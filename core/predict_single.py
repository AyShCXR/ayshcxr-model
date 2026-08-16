# predict_single.py
# AyShCXR — command-line inference. 2026-08-10 rewrite.
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS WAS REWRITTEN
#   The previous version was 620 lines that re-implemented the entire pipeline
#   independently of app.py: its own model loading (searching the working
#   directory for .pth files by name), its own 14-disease list, its own 224px
#   1-channel transform, its own report formatting.
#
#   Once app.py moved to registry-driven loading, 21 findings, calibration,
#   salience ranking and Bayesian fusion, this file silently diverged — the CLI
#   and the web app would give DIFFERENT ANSWERS for the same X-ray. It was also
#   simply broken: it crashed on the ✅ character on Windows terminals, and it
#   looked for checkpoints in the working directory after they moved to models/.
#
#   It is now a thin wrapper over exactly the same modules app.py uses, so the
#   two cannot disagree. ~120 lines instead of 620.
#
# USAGE
#   python core/predict_single.py xray.png
#   python core/predict_single.py xray.png --symptoms fever_high,cough_productive
#   python core/predict_single.py xray.png --json          machine-readable
#   python core/predict_single.py *.png --json > results.jsonl   batch
#   AYSHCXR_MODELS=chexpert_densenet121_v1 python core/predict_single.py x.png
#
# The last form matters for edge work: it runs the single 30MB model, which is
# the deployment candidate, so latency measured here reflects the real device.

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(f"Cannot find .ayshcxr_root above {_HERE}")
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
for _s in (_sys.stdout, _sys.stderr):          # cp1252 consoles cannot encode ✅
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import time
import numpy as np
import torch
from PIL import Image

import model_loader as ml
import calibration as cal
import clinical_extras as cx
import image_quality as iq
import risk_factors as rf
import medical_knowledge_ext as mkext
from symptom_fusion import fuse as bayes_fuse
from disease_ontology import MODEL_REGISTRY


def load_json(path, key=None, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d[key] if key else d
    except Exception:
        return default if default is not None else {}


def analyse(paths, symptoms=None, occupation="", conditions="", smoking="",
            mc_passes=20, model_ids=None):
    """Run the SAME pipeline app.py runs. Yields one result dict per image."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = model_ids or [m.strip() for m in _os.environ.get(
        "AYSHCXR_MODELS",
        "chexpert_densenet121_v1,chexpert_efficientnet_b4_v1,"
        "chexpert_rad_dino_v1,nih_densenet121_v1").split(",") if m.strip()]

    loaded, failed = ml.load_models(ids, dev)
    for mid, why in failed:
        print(f"⚠️  {mid} unavailable — {why}", file=_sys.stderr)
    if not loaded:
        _sys.exit("❌ No models could be loaded.")

    thresholds = load_json("results/disease_thresholds.json", "thresholds", {})
    baselines = load_json("results/finding_baselines.json", "baselines", {})
    risk_lrs = rf.likelihood_ratios(
        rf.parse(occupation=occupation, conditions=conditions, smoking=smoking))
    answered = {k: True for k, v in (symptoms or {}).items() if v}

    for path in paths:
        t0 = time.time()
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            yield {"image": str(path), "error": f"could not open: {e}"}
            continue

        quality = iq.assess(img)
        merged, per_model, unc = ml.predict(loaded, img, dev, mc_passes=mc_passes)
        merged = cal.calibrate_all(merged)

        fused = bayes_fuse(merged, answered, risk_lrs=risk_lrs)
        preds = []
        for d, r in fused.items():
            preds.append({
                "disease": d,
                "probability": round(r["posterior"], 4),
                "image_only": round(r["prior"], 4),
                "uncertainty": round(float(unc.get(d, 0.0)), 4),
                "is_pathology": mkext.is_pathology(d),
            })

        ranked = cx.rank_by_salience(preds, baselines)
        for p in ranked:
            ab = cx.should_abstain(p["probability"], p["uncertainty"], p["disease"])
            p["abstain"] = ab["abstain"]
            p["threshold"] = (thresholds.get(p["disease"]) or {}).get("detected")
            p["flagged"] = (p["threshold"] is not None
                            and p["probability"] >= p["threshold"])

        yield {
            "image": str(path),
            "elapsed_s": round(time.time() - t0, 3),
            "models": [e["id"] for e in loaded],
            "quality": quality["overall"],
            "quality_warnings": [w["check"] for w in quality.get("warnings", [])],
            "study_abstain": cx.panel_abstention(ranked)["abstain"],
            "findings": ranked,
        }


def print_human(res):
    if "error" in res:
        print(f"\n❌ {res['image']}: {res['error']}")
        return
    print("\n" + "=" * 68)
    print(f"  {_os.path.basename(res['image'])}")
    print("=" * 68)
    print(f"  models {len(res['models'])} | {res['elapsed_s']}s | "
          f"film quality: {res['quality']}"
          + (f" ({', '.join(res['quality_warnings'])})"
             if res["quality_warnings"] else ""))
    if res["study_abstain"]:
        print("\n  ⚠️  SYSTEM COULD NOT REACH A CONFIDENT CONCLUSION — "
              "refer for radiologist review")

    path = [f for f in res["findings"] if f["is_pathology"]]
    print(f"\n  {'finding':<28}{'prob':>7}{'±unc':>7}{'salience':>10}   ")
    print("  " + "-" * 60)
    for f in path[:8]:
        mark = " ⚑" if f["flagged"] else "  "
        ab = " ABSTAIN" if f["abstain"] else ""
        print(f"  {f['disease']:<28}{f['probability']:>7.3f}"
              f"{f['uncertainty']:>7.3f}{f['salience']:>10.2f}{mark}{ab}")

    ctx = [f for f in res["findings"] if not f["is_pathology"] and f["flagged"]]
    if ctx:
        print("\n  context (not diagnoses): "
              + ", ".join(f"{c['disease']} {c['probability']:.2f}" for c in ctx))
    print("\n  ⚑ = above this finding's own detection threshold")
    print("  Research prototype — not for standalone clinical diagnosis.")


def main():
    ap = argparse.ArgumentParser(description="AyShCXR command-line inference")
    ap.add_argument("images", nargs="+", help="X-ray image file(s)")
    ap.add_argument("--symptoms", default="",
                    help="comma-separated symptom keys, e.g. fever_high,cough_productive")
    ap.add_argument("--occupation", default="")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--smoking", default="", choices=["", "no", "yes", "past", "heavy"])
    ap.add_argument("--passes", type=int, default=20,
                    help="MC-Dropout passes (0 = fastest, no within-model uncertainty)")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per line, for batch use")
    a = ap.parse_args()

    symptoms = {k.strip(): True for k in a.symptoms.split(",") if k.strip()}
    for res in analyse(a.images, symptoms, a.occupation, a.conditions,
                       a.smoking, a.passes):
        if a.json:
            print(json.dumps(res))
        else:
            print_human(res)


if __name__ == "__main__":
    main()
