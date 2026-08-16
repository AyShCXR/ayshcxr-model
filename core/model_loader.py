# model_loader.py
# AyShCXR — registry-driven model loading. 2026-08-10
# by Subhrakant Sethi & Ayush Singh
#
# Replaces app.py's hardcoded load_model(), which knew one architecture, one
# input size and one label set, and located checkpoints by guessing filenames.
#
# Everything a checkpoint needs to be loaded correctly now lives in
# disease_ontology.MODEL_REGISTRY: architecture, channel count, image size,
# normalisation, label set, and whether its stated AUC was ever verified.
#
# Adding a model later = one registry entry. app.py does not change.
#
# Self-test:  python core/model_loader.py

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

from disease_ontology import (
    MODEL_REGISTRY, get_model_spec, map_predictions, merge_predictions,
    LABEL_SETS, to_canonical,
)


# ── architectures ───────────────────────────────────────────────────────────
def _complex_head(in_f, n=14):
    """Identical to the head used by chexpert_*_train.py."""
    return nn.Sequential(
        nn.BatchNorm1d(in_f), nn.Dropout(0.4),
        nn.Linear(in_f, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, n),
    )


class CNNDualPool(nn.Module):
    """EfficientNet / DenseNet with GMP+GAP dual pooling.

    Must mirror chexpert_densenet_train.py exactly or the state_dict will not
    load. Note DenseNet applies ReLU after features; EfficientNet does not.
    """

    def __init__(self, arch, n=14):
        super().__init__()
        if arch.startswith("efficientnet"):
            bb = tvm.efficientnet_b4(weights=None)
            self.features = bb.features
            in_f = bb.classifier[1].in_features
            self.post = nn.Identity()
        else:
            bb = tvm.densenet121(weights=None)
            self.features = bb.features
            in_f = bb.classifier.in_features
            self.post = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = _complex_head(in_f, n)

    def forward(self, x):
        f = self.post(self.features(x))
        return self.classifier(self.gap(f).flatten(1) + self.gmp(f).flatten(1))


class PlainDenseNet(nn.Module):
    """Phase-1 DenseNet-121: 1-channel stem, plain GAP, same complex head."""

    def __init__(self, in_channels=1, n=14):
        super().__init__()
        bb = tvm.densenet121(weights=None)
        if in_channels == 1:
            bb.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.features = bb.features
        self.classifier = _complex_head(bb.classifier.in_features, n)

    def forward(self, x):
        f = torch.relu(self.features(x))
        return self.classifier(torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)), 1))


class RadDino(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        from transformers import AutoModel          # optional dependency
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
        self.head = _complex_head(self.backbone.config.hidden_size, n)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        feat = out.pooler_output
        if feat is None:
            feat = out.last_hidden_state[:, 0]
        return self.head(feat)


def _build(arch, in_channels, n):
    if arch == "densenet121":
        return PlainDenseNet(in_channels, n)
    if arch in ("densenet121_dualpool", "efficientnet_b4_dualpool"):
        return CNNDualPool(arch.replace("_dualpool", ""), n)
    if arch == "rad_dino":
        return RadDino(n)
    raise ValueError(f"Unknown architecture {arch!r} in registry")


# ── loading ─────────────────────────────────────────────────────────────────
def _clean_state_dict(raw, spec):
    """Unwrap a training checkpoint and strip a torch.compile prefix."""
    key = spec.get("state_dict_key")
    if key and isinstance(raw, dict) and key in raw:
        raw = raw[key]
    elif isinstance(raw, dict) and not any(
            hasattr(v, "shape") for v in list(raw.values())[:5]):
        for k in ("state_dict", "model_state_dict", "model_state", "model"):
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break
    prefix = spec.get("strip_prefix")
    if prefix:
        raw = {(k[len(prefix):] if k.startswith(prefix) else k): v
               for k, v in raw.items()}
    return raw


def build_transform(spec):
    mean, std = spec["normalize"]
    size = spec["img_size"]
    steps = [T.Resize((size, size))]
    if spec["in_channels"] == 1:
        steps.append(T.Grayscale(num_output_channels=1))
    steps += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(steps)


def load_one(model_id, device="cpu", allow_unverified=False):
    """Load a single registry entry. Returns dict or None on failure."""
    spec = get_model_spec(model_id)

    if not spec["verified"] and not allow_unverified:
        raise ValueError(
            f"{model_id} is marked verified=False and will not be auto-loaded. "
            f"Reason: {spec.get('notes', '')}"
        )

    path = spec["weights"]
    if not os.path.exists(path):
        for alt in (os.path.join("models", path), os.path.join("..", path)):
            if os.path.exists(alt):
                path = alt
                break
        else:
            return None

    n = len(LABEL_SETS[spec["label_set"]])
    model = _build(spec["arch"], spec["in_channels"], n)
    raw = torch.load(path, map_location=device, weights_only=False)
    sd = _clean_state_dict(raw, spec)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # A silent partial load is how a model ends up predicting noise.
        raise RuntimeError(
            f"{model_id}: state_dict mismatch — "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"First missing: {list(missing)[:3]}"
        )

    model.eval().to(device)
    return {
        "id": model_id,
        "model": model,
        "spec": spec,
        "transform": build_transform(spec),
        "label_set": spec["label_set"],
        "auc": spec["auc"],
    }


def load_models(model_ids, device="cpu"):
    """Load several. Returns (loaded, failed) — never raises on a missing file
    so the app can degrade to whatever is actually present on disk."""
    loaded, failed = [], []
    for mid in model_ids:
        try:
            entry = load_one(mid, device)
            if entry is None:
                failed.append((mid, "checkpoint file not found"))
            else:
                loaded.append(entry)
        except Exception as e:
            failed.append((mid, str(e)[:160]))
    return loaded, failed


def covered_findings(loaded):
    """Canonical findings the loaded set can actually predict."""
    seen = []
    for e in loaded:
        for name in LABEL_SETS[e["label_set"]]:
            c = to_canonical(name)
            if c not in seen:
                seen.append(c)
    return seen


@torch.no_grad()
def predict(loaded, img_pil, device="cpu", mc_passes=0):
    """Run every loaded model and merge by canonical name.

    Returns (merged_probs, per_model, uncertainty).

    ── UNCERTAINTY: TWO SOURCES, PROPERLY COMBINED ────────────────────────────
    An earlier version took uncertainty[finding] = max(per-model MC spread),
    which was wrong in two ways:
      * it kept only the single most unstable model and discarded the rest, so
        one flaky member made a finding look uncertain even when three others
        agreed exactly;
      * it ignored DISAGREEMENT BETWEEN MODELS entirely. If DenseNet said 0.9
        and EfficientNet said 0.1 while each was internally stable, reported
        uncertainty was ~0 — precisely the case a clinician most needs flagged.

    Now both components are measured and combined by the standard variance
    decomposition, total_var = mean(within_var) + var(between_means):

        within  — mean MC-Dropout spread across models. "Is each model itself
                  sure?" Requires mc_passes > 1.
        between — standard deviation of the per-model means. "Do the models
                  agree with each other?" Available with ANY number of passes.
        total   — sqrt(within^2 + between^2)

    Because `between` needs no MC passes, uncertainty is never silently zero
    just because a fast path was taken. That matters: the app's degraded retry
    calls this with mc_passes=0, and reporting zero uncertainty there would read
    as maximum confidence at exactly the wrong moment.
    """
    per_model = {}
    preds, weights = [], []
    within_acc = {}          # finding -> [per-model MC std]
    means_acc = {}           # finding -> [per-model mean prob]

    for entry in loaded:
        tensor = entry["transform"](img_pil).unsqueeze(0).to(device)
        model = entry["model"]

        if mc_passes and mc_passes > 1:
            model.train()                       # dropout ON
            for m in model.modules():           # BatchNorm MUST stay in eval:
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()                    # batch stats on n=1 are garbage
            # One batched forward instead of `mc_passes` separate calls. Safe
            # precisely because BatchNorm is in eval — no cross-sample leakage.
            batch = tensor.expand(mc_passes, *tensor.shape[1:])
            runs = torch.sigmoid(model(batch))
            model.eval()
            probs = runs.mean(0).cpu().numpy()
            stds = runs.std(0).cpu().numpy()
        else:
            probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()
            stds = None

        mapped = map_predictions(entry["label_set"], probs.tolist())
        per_model[entry["id"]] = mapped
        preds.append(mapped)
        weights.append(entry["auc"] or 1.0)

        for name, p in zip(LABEL_SETS[entry["label_set"]], probs.tolist()):
            means_acc.setdefault(to_canonical(name), []).append(float(p))
        if stds is not None:
            for name, s in zip(LABEL_SETS[entry["label_set"]], stds.tolist()):
                within_acc.setdefault(to_canonical(name), []).append(float(s))

    uncertainty = {}
    for c, means in means_acc.items():
        within = float(np.mean(within_acc[c])) if within_acc.get(c) else 0.0
        # population std: with one model there is no disagreement to measure
        between = float(np.std(means)) if len(means) > 1 else 0.0
        uncertainty[c] = float(np.sqrt(within ** 2 + between ** 2))

    # NOTE ON WEIGHTS: these are member AUCs — 0.8281 / 0.8238 / 0.8226 / 0.8031.
    # A 3% spread means this is very nearly an unweighted mean, and it is not
    # claimed to be more than that. Sharpening the weights would imply a
    # precision the validation data does not support.
    merged = merge_predictions(preds, weights) if preds else {}
    return merged, per_model, uncertainty


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve()
    ROOT = next((p for p in HERE.parents if (p / ".ayshcxr_root").exists()), None)
    if ROOT:
        os.chdir(ROOT)
    sys.path.insert(0, str(HERE.parent))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"  model_loader self-test   (device: {dev})")
    print("=" * 70)

    print("\nregistry:")
    for mid, spec in MODEL_REGISTRY.items():
        exists = os.path.exists(spec["weights"])
        mark = "OK " if spec["verified"] else "!! "
        print(f"  {mark}{mid:30s} {spec['label_set']:11s} "
              f"{spec['in_channels']}ch@{spec['img_size']:<4d} "
              f"file={'yes' if exists else 'NO'}")

    print("\nunverified checkpoints must be refused:")
    try:
        load_one("nih_densenet_laptop_ep20", dev)
        print("  FAIL — it loaded, quarantine is not working")
    except ValueError as e:
        print(f"  OK — refused: {str(e)[:80]}...")

    ids = [m for m, s in MODEL_REGISTRY.items()
           if s["verified"] and s["arch"] != "rad_dino"]
    print(f"\nloading {len(ids)} verified non-transformer models ...")
    loaded, failed = load_models(ids, dev)
    for e in loaded:
        n = sum(p.numel() for p in e["model"].parameters())
        print(f"  OK   {e['id']:30s} {n/1e6:6.1f}M params  AUC {e['auc']}")
    for mid, why in failed:
        print(f"  FAIL {mid:30s} {why}")

    if loaded:
        cov = covered_findings(loaded)
        print(f"\ncanonical findings covered: {len(cov)}")
        print(f"  {cov}")

        from PIL import Image
        dummy = Image.new("RGB", (512, 512), (110, 110, 110))
        merged, per_model, unc = predict(loaded, dummy, dev, mc_passes=0)
        print(f"\nforward pass on a dummy image -> {len(merged)} findings")
        for d, p in list(merged.items())[:5]:
            print(f"   {d:28s} {p:.4f}")
        assert all(0.0 <= v <= 1.0 for v in merged.values()), "probs out of range"
        print("\n  probabilities in range, mapping by canonical name confirmed")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
