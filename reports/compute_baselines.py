# compute_baselines.py
# AyShCXR — per-finding baseline distributions for salience ranking. 2026-08-10
#
# WHY THIS EXISTS SEPARATELY
#   The first baselines were measured on 500 random VinDr images while the
#   thresholds were later measured on NIH films. Salience then compared an NIH
#   score against a VinDr distribution — different populations, different
#   scanners, different disease prevalence.
#
#   Measured consequence: Edema's stored p90 was 0.096, but its real p90 on
#   healthy NIH films is 0.228. Salience for Edema was therefore inflated on
#   every NIH image, and it stole the top slot on Cardiomegaly, Effusion and
#   Pneumonia films repeatedly.
#
#   Baselines and thresholds must come from the SAME population. This script
#   computes them on NIH films with no reported finding, matching
#   tune_thresholds.py.
#
# MINIMUM SPREAD GUARD
#   Salience = (p - median) / (p90 - median). If a finding is nearly constant
#   that denominator approaches zero and trivial noise produces enormous
#   salience. Hernia's spread is 0.013 and Pleural Thickening's is 0.018 — both
#   effectively constants. A floor on the denominator stops a finding that
#   carries no information from dominating the ranking.
#
#   python reports/compute_baselines.py [n] [--write]

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import io, json, time, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
import torch

with contextlib.redirect_stdout(io.StringIO()):
    import model_loader as ml, calibration as cal
    from disease_ontology import MODEL_REGISTRY, CANONICAL_ORDER

NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
       "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
       "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]

# Findings whose measured spread falls below this are treated as uninformative
# for ranking. 0.05 is roughly a third of a typical finding's spread.
MIN_SPREAD = 0.05


def main():
    n = int(_sys.argv[1]) if len(_sys.argv) > 1 and _sys.argv[1].isdigit() else 400
    write = "--write" in _sys.argv

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        loaded, _ = ml.load_models(
            [m for m, s in MODEL_REGISTRY.items() if s["verified"]], dev)

    df = pd.read_csv("data/nih_full_labels.csv")
    df["n_find"] = df[NIH].sum(axis=1)
    healthy = df[df.n_find == 0].sample(min(n, int((df.n_find == 0).sum())),
                                        random_state=5)

    print("=" * 78)
    print("  Baseline distributions — measured on films with NO reported finding")
    print("=" * 78)
    print(f"\n  {len(healthy)} healthy NIH films (same population as the thresholds)\n")

    t0 = time.time()
    rows = []
    for i, (_, r) in enumerate(healthy.iterrows()):
        try:
            img = Image.open(r["full_path"]).convert("RGB")
        except Exception:
            continue
        m, _, _ = ml.predict(loaded, img, dev, mc_passes=0)
        rows.append(cal.calibrate_all(m))
        if (i + 1) % 150 == 0:
            print(f"    {i+1}/{len(healthy)}  ({time.time()-t0:.0f}s)")
    P = pd.DataFrame(rows)
    print(f"  done ({time.time()-t0:.0f}s)\n")

    try:
        old = json.load(open("results/finding_baselines.json"))["baselines"]
    except Exception:
        old = {}

    stats, flat = {}, []
    print(f"{'finding':<28}{'median':>8}{'p90':>8}{'spread':>8}{'used':>8}"
          f"{'old p90':>9}  note")
    print("-" * 78)
    for c in CANONICAL_ORDER:
        if c not in P.columns:
            continue
        v = P[c].values
        med = float(np.median(v))
        p90 = float(np.percentile(v, 90))
        spread = p90 - med
        used = max(spread, MIN_SPREAD)
        note = ""
        if spread < MIN_SPREAD:
            note = "NEAR-CONSTANT — spread floored, ranking damped"
            flat.append(c)
        o = old.get(c, {}).get("p90")
        os_ = f"{o:.3f}" if o is not None else "  -  "
        stats[c] = {"mean": float(v.mean()), "median": med, "p90": p90,
                    "std": float(v.std()), "spread": spread,
                    "effective_spread": used}
        print(f"{c:<28}{med:>8.3f}{p90:>8.3f}{spread:>8.3f}{used:>8.3f}"
              f"{os_:>9}  {note}")
    print("-" * 78)
    if flat:
        print(f"  {len(flat)} finding(s) are effectively constant on healthy films:")
        print(f"    {', '.join(flat)}")
        print("  These carry almost no information. The spread floor stops them")
        print("  dominating the ranking through noise alone.")

    if write:
        json.dump({"n_images": len(P),
                   "source": "NIH ChestX-ray14, films with no reported finding",
                   "note": "Measured on the SAME population as the thresholds. "
                           "effective_spread is max(spread, %.2f) — see "
                           "MIN_SPREAD." % MIN_SPREAD,
                   "min_spread": MIN_SPREAD,
                   "baselines": stats},
                  open("results/finding_baselines.json", "w"), indent=1)
        print(f"\n  WROTE results/finding_baselines.json")
    else:
        print(f"\n  (dry run — re-run with --write to apply)")


if __name__ == "__main__":
    main()
