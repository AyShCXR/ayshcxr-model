# disease_ontology.py
# AyShCXR — canonical disease vocabulary, per-dataset label mappings, model registry.
# by Subhrakant Sethi & Ayush Singh
#
# PURPOSE
#   Different datasets name and order their labels differently. NIH says "Effusion",
#   CheXpert says "Pleural Effusion". Both emit 14 logits, but slot 3 means
#   "Infiltration" in one and "Lung Lesion" in the other. Nothing in the tensor
#   says which is which, so a mismatched checkpoint loads WITHOUT error and then
#   silently mislabels every prediction.
#
#   This module makes that class of bug impossible: predictions are converted to
#   {canonical_name: probability} at the model boundary, and everything downstream
#   is keyed by name, never by index.
#
# TO ADD A NEW DATASET LATER
#   1. add any genuinely new findings to CANONICAL_FINDINGS
#   2. add its ordered label list to LABEL_SETS
#   3. add its checkpoint to MODEL_REGISTRY
#   app.py does not change.
#
# Self-test:  python disease_ontology.py

from typing import Dict, List, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# 1. CANONICAL FINDINGS  —  union of NIH-14 and CheXpert-14, deduplicated (21)
# ─────────────────────────────────────────────────────────────────────────────
# kind:
#   "disease" — a pathology; gets a report card, urgency score, treatment plan
#   "device"  — hardware visible in the image (tube/line/pacemaker). NOT a
#               pathology: no urgency, no treatment. Clinically useful context.
#   "normal"  — explicit absence of findings. Never triaged, never treated.
#
# source: which dataset(s) supply a ground-truth label for this finding.

CANONICAL_FINDINGS: Dict[str, dict] = {
    # ---- present in BOTH datasets (7) ------------------------------------
    "Atelectasis":      {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},
    "Cardiomegaly":     {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},
    "Effusion":         {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},
    "Pneumonia":        {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},
    "Pneumothorax":     {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},
    "Consolidation":    {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},
    "Edema":            {"kind": "disease", "source": ["NIH14", "CHEXPERT14"]},

    # ---- NIH only (7) ----------------------------------------------------
    "Infiltration":     {"kind": "disease", "source": ["NIH14"]},
    "Mass":             {"kind": "disease", "source": ["NIH14"]},
    "Nodule":           {"kind": "disease", "source": ["NIH14"]},
    "Emphysema":        {"kind": "disease", "source": ["NIH14"]},
    "Fibrosis":         {"kind": "disease", "source": ["NIH14"]},
    "Pleural Thickening": {"kind": "disease", "source": ["NIH14"]},
    "Hernia":           {"kind": "disease", "source": ["NIH14"]},

    # ---- CheXpert only (7) -----------------------------------------------
    "Enlarged Cardiomediastinum": {"kind": "disease", "source": ["CHEXPERT14"]},
    "Lung Opacity":     {"kind": "disease", "source": ["CHEXPERT14"]},
    "Lung Lesion":      {"kind": "disease", "source": ["CHEXPERT14"]},
    "Pleural Other":    {"kind": "disease", "source": ["CHEXPERT14"]},
    "Fracture":         {"kind": "disease", "source": ["CHEXPERT14"]},
    "Support Devices":  {"kind": "device",  "source": ["CHEXPERT14"]},
    "No Finding":       {"kind": "normal",  "source": ["CHEXPERT14"]},
}

CANONICAL_ORDER: List[str] = list(CANONICAL_FINDINGS.keys())          # 21
DISEASE_FINDINGS: List[str] = [k for k, v in CANONICAL_FINDINGS.items()
                               if v["kind"] == "disease"]             # 19

# ─────────────────────────────────────────────────────────────────────────────
# 2. ALIASES  —  dataset spelling  ->  canonical name
# ─────────────────────────────────────────────────────────────────────────────
# "Effusion" is canonical (not "Pleural Effusion") because medical_knowledge.py
# already keys on it — choosing the other way would force edits across ~3,500
# lines for no benefit.

ALIASES: Dict[str, str] = {
    "pleural effusion":   "Effusion",
    "effusion":           "Effusion",
    "pleural_thickening": "Pleural Thickening",
    "pleural thickening": "Pleural Thickening",
    "enlarged cardiom":   "Enlarged Cardiomediastinum",
    "no finding":         "No Finding",
    "support devices":    "Support Devices",
    "lung opacity":       "Lung Opacity",
    "lung lesion":        "Lung Lesion",
    "pleural other":      "Pleural Other",
}


def to_canonical(name: str) -> str:
    """Normalise any dataset spelling to its canonical form.

    Raises KeyError for genuinely unknown findings — callers that must not
    crash should use `safe_to_canonical` instead.
    """
    n = name.strip()
    if n in CANONICAL_FINDINGS:
        return n
    key = n.lower().replace("_", " ")
    if key in ALIASES:
        return ALIASES[key]
    for c in CANONICAL_FINDINGS:
        if c.lower() == key:
            return c
    raise KeyError(f"Unknown finding: {name!r}")


def safe_to_canonical(name: str):
    """Like to_canonical but returns None instead of raising."""
    try:
        return to_canonical(name)
    except KeyError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL SETS  —  ORDER IS LOAD-BEARING
# ─────────────────────────────────────────────────────────────────────────────
# These lists must match each model's output order EXACTLY. Getting one wrong
# silently mislabels every prediction, so both are transcribed verbatim from
# the code that trained them:
#   NIH14      <- app.py                     DISEASES  (line 37)
#   CHEXPERT14 <- chexpert_densenet_train.py TARGET_LABELS (line 21)

LABEL_SETS: Dict[str, List[str]] = {
    "NIH14": [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural Thickening", "Hernia",
    ],
    "CHEXPERT14": [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices", "No Finding",
    ],
}


def map_predictions(label_set: str, probs: Sequence[float]) -> Dict[str, float]:
    """Convert a model's raw output vector into {canonical_name: probability}.

    This is THE boundary. Nothing downstream should ever index a prediction
    tensor by position again.
    """
    if label_set not in LABEL_SETS:
        raise KeyError(f"Unknown label set {label_set!r}. "
                       f"Known: {list(LABEL_SETS)}")
    names = LABEL_SETS[label_set]
    if len(probs) != len(names):
        raise ValueError(
            f"{label_set} expects {len(names)} outputs, got {len(probs)}. "
            "This usually means the checkpoint does not match the declared "
            "label_set in MODEL_REGISTRY."
        )
    return {to_canonical(n): float(p) for n, p in zip(names, probs)}


def merge_predictions(preds: Sequence[Dict[str, float]],
                      weights: Sequence[float] = None) -> Dict[str, float]:
    """Merge several models' canonical predictions into one.

    Findings predicted by multiple models are averaged (optionally weighted);
    findings only one model knows about pass through unchanged. This is what
    gives 21-disease coverage from models that each only know 14.
    """
    if weights is None:
        weights = [1.0] * len(preds)
    if len(weights) != len(preds):
        raise ValueError("weights and preds must be the same length")

    acc: Dict[str, float] = {}
    wsum: Dict[str, float] = {}
    for p, w in zip(preds, weights):
        for name, prob in p.items():
            acc[name] = acc.get(name, 0.0) + prob * w
            wsum[name] = wsum.get(name, 0.0) + w
    merged = {n: acc[n] / wsum[n] for n in acc}
    return {n: merged[n] for n in CANONICAL_ORDER if n in merged}


def coverage(label_sets: Sequence[str]) -> Dict[str, List[str]]:
    """Which canonical findings are covered / missing for a set of models."""
    covered = set()
    for ls in label_sets:
        covered |= {to_canonical(n) for n in LABEL_SETS[ls]}
    return {
        "covered": [c for c in CANONICAL_ORDER if c in covered],
        "missing": [c for c in CANONICAL_ORDER if c not in covered],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL REGISTRY  —  every checkpoint declares itself
# ─────────────────────────────────────────────────────────────────────────────
# `verified` means: this file's ACTUAL weights were evaluated and produced the
# stated AUC. A checkpoint whose filename claims a score its weights were never
# measured at is verified=False and must never be auto-loaded.
#
# arch values:
#   "densenet121"          — torchvision densenet121, plain GAP  (Phase 1)
#   "densenet121_dualpool" — CNNDualPool wrapper, GMP+GAP        (CheXpert)
#   "efficientnet_b4_dualpool"
#   "rad_dino"

MODEL_REGISTRY: Dict[str, dict] = {
    "nih_densenet121_v1": {
        "weights":      "densenet121_BEST_auc0.8031_ep19_SAFE.pth",
        "arch":         "densenet121",
        "in_channels":  1,
        "img_size":     224,
        "normalize":    ([0.485], [0.229]),
        "label_set":    "NIH14",
        "auc":          0.8031,
        "verified":     True,
        "state_dict_key": None,        # file IS the state_dict
        "strip_prefix":   None,
        "trained_on":   "NIH ChestX-ray14",
        "notes":        "Phase-1 model. Currently what app.py serves.",
    },

    "chexpert_densenet121_v1": {
        "weights":      "chexpert_densenet121_best_auc0.8281_ep12.pth",
        "arch":         "densenet121_dualpool",
        "in_channels":  3,
        "img_size":     380,
        "normalize":    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "label_set":    "CHEXPERT14",
        "auc":          0.8281,
        "verified":     True,
        "state_dict_key": None,
        "strip_prefix":   "_orig_mod.",   # saved from a torch.compile'd model
        "trained_on":   "CheXpert Plus (impression_fixed labels)",
        "notes":        "Best single model. 30MB — the edge-deployment candidate.",
    },

    "chexpert_efficientnet_b4_v1": {
        "weights":      "chexpert_efficientnet_b4_best_auc0.8226_ep16.pth",
        "arch":         "efficientnet_b4_dualpool",
        "in_channels":  3,
        "img_size":     380,
        "normalize":    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "label_set":    "CHEXPERT14",
        "auc":          0.8226,
        "verified":     True,
        "state_dict_key": None,
        "strip_prefix":   "_orig_mod.",
        "trained_on":   "CheXpert Plus (impression_fixed labels)",
        "notes":        "75MB.",
    },

    "chexpert_rad_dino_v1": {
        "weights":      "chexpert_rad_dino_best_auc0.8238_ep4.pth",
        "arch":         "rad_dino",
        "in_channels":  3,
        "img_size":     224,
        "normalize":    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "label_set":    "CHEXPERT14",
        "auc":          0.8238,
        "verified":     True,
        "state_dict_key": None,
        "strip_prefix":   "_orig_mod.",
        "trained_on":   "CheXpert Plus (impression_fixed labels)",
        "notes":        "348MB ViT. Needs transformers==4.57.6 + "
                        "huggingface_hub==0.36.2. NOT edge-deployable.",
    },

    # ---- present on disk but deliberately NOT auto-loadable ---------------
    "nih_densenet_laptop_ep20": {
        "weights":      "densenet_laptop_auc0.8138_ep18peak_ckpt20_SAFE.pth",
        "arch":         "densenet121",
        "in_channels":  1,
        "img_size":     224,
        "normalize":    ([0.485], [0.229]),
        "label_set":    "NIH14",
        "auc":          None,
        "verified":     False,
        "state_dict_key": "model_state",   # full training checkpoint
        "strip_prefix":   None,
        "trained_on":   "NIH ChestX-ray14",
        "notes":        "FILENAME IS MISLEADING. The 0.8138 peak was at epoch 18; "
                        "these weights are from epoch 20 and were never scored. "
                        "Evaluate before trusting. Do not auto-load.",
    },
}


def get_model_spec(model_id: str) -> dict:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_id!r}. "
                       f"Known: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_id]


def verified_models(label_set: str = None) -> List[str]:
    """Model ids safe to auto-load, best AUC first."""
    ids = [k for k, v in MODEL_REGISTRY.items()
           if v["verified"] and (label_set is None or v["label_set"] == label_set)]
    return sorted(ids, key=lambda k: MODEL_REGISTRY[k]["auc"] or 0, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 68)
    print("  AyShCXR — disease ontology self-test")
    print("=" * 68)

    print(f"\ncanonical findings : {len(CANONICAL_ORDER)}")
    for kind in ("disease", "device", "normal"):
        names = [k for k, v in CANONICAL_FINDINGS.items() if v["kind"] == kind]
        print(f"  {kind:8s} ({len(names):2d}) : {', '.join(names)}")

    # every dataset label must resolve
    print("\nlabel-set resolution:")
    for ls, names in LABEL_SETS.items():
        bad = [n for n in names if safe_to_canonical(n) is None]
        print(f"  {ls:12s} {len(names)} labels -> "
              f"{'OK' if not bad else 'UNRESOLVED: ' + str(bad)}")
        assert not bad, f"{ls} has unmapped labels: {bad}"

    # NIH and CheXpert must agree on exactly 7 findings
    nih = {to_canonical(n) for n in LABEL_SETS["NIH14"]}
    chex = {to_canonical(n) for n in LABEL_SETS["CHEXPERT14"]}
    shared = nih & chex
    print(f"\nshared by both datasets ({len(shared)}): {sorted(shared)}")
    print(f"NIH only  ({len(nih - chex)}): {sorted(nih - chex)}")
    print(f"CheXpert only ({len(chex - nih)}): {sorted(chex - nih)}")
    assert len(shared) == 7, f"expected 7 shared, got {len(shared)}"
    assert nih | chex == set(CANONICAL_ORDER), "union != canonical list"

    # mapping demo — deliberately different orders
    nih_probs  = [0.9, 0.1, 0.2, 0.3, 0.05, 0.05, 0.8, 0.02,
                  0.4, 0.3, 0.01, 0.01, 0.02, 0.001]
    chex_probs = [0.2, 0.15, 0.6, 0.05, 0.35, 0.45, 0.7,
                  0.85, 0.03, 0.25, 0.01, 0.02, 0.9, 0.05]
    m_nih  = map_predictions("NIH14", nih_probs)
    m_chex = map_predictions("CHEXPERT14", chex_probs)

    print("\nposition-independence check (Pneumonia sits at different indices):")
    print(f"  NIH14      index 6  -> Pneumonia = {m_nih['Pneumonia']}")
    print(f"  CHEXPERT14 index 6  -> Pneumonia = {m_chex['Pneumonia']}")
    assert m_nih["Pneumonia"] == 0.8 and m_chex["Pneumonia"] == 0.7

    print("\nalias check: CheXpert 'Pleural Effusion' -> canonical 'Effusion'")
    print(f"  value = {m_chex['Effusion']}")
    assert m_chex["Effusion"] == 0.25

    merged = merge_predictions([m_nih, m_chex])
    print(f"\nmerged coverage: {len(merged)} findings from two 14-class models")
    print(f"  Pneumonia (both, averaged) : {merged['Pneumonia']:.3f}"
          f"   [expected {(0.8 + 0.7) / 2:.3f}]")
    print(f"  Hernia    (NIH only)       : {merged['Hernia']:.3f}")
    print(f"  Fracture  (CheXpert only)  : {merged['Fracture']:.3f}")
    assert abs(merged["Pneumonia"] - 0.75) < 1e-9
    assert len(merged) == 21

    cov = coverage(["NIH14", "CHEXPERT14"])
    print(f"\ncoverage with both models: {len(cov['covered'])}/21 "
          f"| missing: {cov['missing'] or 'none'}")

    print("\nregistry:")
    for mid, spec in MODEL_REGISTRY.items():
        mark = "OK " if spec["verified"] else "!! "
        auc = f"{spec['auc']:.4f}" if spec["auc"] else "UNVERIFIED"
        print(f"  {mark}{mid:30s} {spec['label_set']:11s} {auc:>10s}  "
              f"{spec['in_channels']}ch@{spec['img_size']}")
    print(f"\nauto-loadable, best first: {verified_models()}")

    print("\n" + "=" * 68)
    print("  ALL CHECKS PASSED")
    print("=" * 68)
