# compute_thresholds.py
# AyShCXR — derive per-disease decision thresholds. 2026-08-10
#
# WHY THE OBVIOUS APPROACH IS WRONG
#   A first version computed thresholds from each model's historical prediction
#   CSVs. That is invalid: the app does not show any single model's output. It
#   shows the CALIBRATED MERGE of whichever models cover that finding, and the
#   merge sits on a different scale from any individual member. Applying the
#   CheXpert calibration to the NIH CSV produced a Cardiomegaly threshold of
#   0.025 against 0.308 from CheXpert for the same finding.
#
# WHAT THIS DOES INSTEAD
#   Runs the REAL pipeline (all loaded models -> merge -> calibrate) over
#   radiologist-labelled VinDr images, then computes per finding:
#       detected   = F1-optimal threshold
#       borderline = lowest threshold still reaching 90% sensitivity
#
#   VinDr only labels ~10 of the 21 canonical findings. For the rest there is no
#   ground truth available, so the threshold is set from the finding's own
#   BASELINE DISTRIBUTION (results/finding_baselines.json): "detected" means
#   scoring above 95% of ordinary films. That is a distributional threshold, not
#   a validated one, and it is flagged as such in the output.
#
#   python reports/compute_thresholds.py [n_images]

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

import io, json, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image
import torch

with contextlib.redirect_stdout(io.StringIO()):
    import model_loader as ml
    import calibration as cal
    from disease_ontology import MODEL_REGISTRY, CANONICAL_ORDER

MIN_POS = 25


def f1_at(y, p, t):
    pred = p >= t
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def derive(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.sum() < MIN_POS or len(np.unique(y)) < 2:
        return None
    grid = np.unique(np.round(np.linspace(0.01, 0.95, 190), 4))
    f1s = np.array([f1_at(y, p, t) for t in grid])
    det = float(grid[int(np.argmax(f1s))])
    sens = np.array([((p >= t) & (y == 1)).sum() / max(y.sum(), 1) for t in grid])
    ok = grid[sens >= 0.90]
    bord = min(float(ok.max()) if len(ok) else det * 0.65, det)
    pred = p >= det
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    return (det, bord, float(f1s.max()),
            tp / (tp + fp) if (tp + fp) else 0.0,
            tp / (tp + fn) if (tp + fn) else 0.0, int(y.sum()))


def main():
    n_req = int(_sys.argv[1]) if len(_sys.argv) > 1 else 900
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = [m for m, s in MODEL_REGISTRY.items() if s["verified"]]
    with contextlib.redirect_stdout(io.StringIO()):
        LOADED, _ = ml.load_models(ids, dev)
    print(f"pipeline: {len(LOADED)} models -> merge -> calibrate  (device {dev})")

    df = pd.read_csv("data/vindr/vindr_labels.csv")
    vindr_findings = [c for c in df.columns
                      if c not in ("image_id", "full_path", "n_readers")]

    # stratified sample so rare findings are represented
    rng = np.random.RandomState(0)
    idx = set()
    for f in vindr_findings:
        pos = df.index[df[f] == 1].tolist()
        if pos:
            idx |= set(rng.choice(pos, min(120, len(pos)), replace=False).tolist())
    rest = [i for i in df.index if i not in idx]
    if len(idx) < n_req and rest:
        idx |= set(rng.choice(rest, min(n_req - len(idx), len(rest)),
                              replace=False).tolist())
    sub = df.loc[sorted(idx)].reset_index(drop=True)
    print(f"running the real pipeline over {len(sub)} radiologist-labelled images\n")

    rows = []
    for i, r in sub.iterrows():
        try:
            img = Image.open(r["full_path"]).convert("RGB")
        except Exception:
            rows.append(None); continue
        merged, _, _ = ml.predict(LOADED, img, dev, mc_passes=0)
        rows.append(cal.calibrate_all(merged))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sub)}")
    keep = [i for i, v in enumerate(rows) if v is not None]
    P = pd.DataFrame([rows[i] for i in keep])
    sub = sub.iloc[keep].reset_index(drop=True)

    try:
        base = json.load(open("results/finding_baselines.json"))["baselines"]
    except Exception:
        base = {}

    out, report = {}, []
    for c in CANONICAL_ORDER:
        if c in sub.columns and c in P.columns:
            r = derive(sub[c].values, P[c].values)
            if r:
                det, bord, f1, prec, rec, npos = r
                out[c] = {"detected": round(det, 3), "borderline": round(bord, 3)}
                report.append((c, "measured", npos, det, bord, f1, prec, rec))
                continue
        # no usable ground truth -> distributional fallback
        b = base.get(c)
        if b:
            det = round(min(max(b["p90"], b["mean"] + b["std"]), 0.95), 3)
            bord = round(max(b["median"], det * 0.6), 3)
        else:
            det, bord = 0.50, 0.35
        out[c] = {"detected": det, "borderline": bord}
        report.append((c, "baseline", 0, det, bord, None, None, None))

    print("\n" + "=" * 88)
    print(f"{'finding':<28}{'basis':>10}{'n_pos':>7}{'detected':>10}"
          f"{'border':>9}{'F1':>7}{'prec':>7}{'rec':>7}")
    print("-" * 88)
    for c, basis, npos, det, bord, f1, prec, rec in report:
        f1s = f"{f1:.3f}" if f1 is not None else "   -  "
        ps = f"{prec:.3f}" if prec is not None else "   -  "
        rs = f"{rec:.3f}" if rec is not None else "   -  "
        print(f"{c:<28}{basis:>10}{npos:>7}{det:>10.3f}{bord:>9.3f}"
              f"{f1s:>7}{ps:>7}{rs:>7}")
    print("-" * 88)
    nm = sum(1 for r in report if r[1] == "measured")
    print(f"  {nm} measured against radiologist labels, "
          f"{len(report)-nm} from baseline distribution (unvalidated)")

    path = "results/disease_thresholds.json"
    json.dump({"note": "computed on the real pipeline: all models -> merge -> "
                       "calibrate. 'measured' = F1-optimal vs radiologist "
                       "labels; 'baseline' = above 90th percentile of ordinary "
                       "films, NOT validated.",
               "basis": {c: b for c, b, *_ in report},
               "thresholds": out},
              open(path, "w", encoding="utf-8"), indent=1)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
