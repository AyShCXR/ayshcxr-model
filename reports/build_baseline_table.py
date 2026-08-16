# build_baseline_table.py  —  RUN ON YOUR LAPTOP (no GPU, no images needed)
# Turns cv_results.csv (after cv_reference_baseline.py has added the vanilla DenseNet row)
# into the head-to-head table Sir asked for: your models vs. the reference CheXpert-paper
# DenseNet-121, same 3-fold CV, same 90k sample, test set untouched.
#   python build_baseline_table.py
# needs: pandas   (pip install pandas)


# ── AyShCXR path bootstrap (added 2026-08-08 during folder reorganisation) ──
# This script now lives in a subfolder but still refers to data files by bare
# name (e.g. "nih_full_labels.csv"). Pointing the working directory at the
# project root keeps every existing relative path working unchanged, and puts
# core/ on sys.path so `import medical_knowledge` still resolves.
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(
        "Could not locate the AyShCXR project root: no .ayshcxr_root marker "
        f"found above {_HERE}. Restore that file or run from the project root."
    )
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ── end path bootstrap ──────────────────────────────────────────────────────

import os, sys, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")   # avoid Windows cp1252 console crashes
except Exception: pass

RESULTS = "cv_results.csv"
OUT     = "Baseline_vs_Ours.csv"

# arch key -> (human label, role).  Reference = the CheXpert-paper model; Ours = your work.
META = {
    "densenet121_vanilla": ("DenseNet-121 (CheXpert paper, Irvin & Rajpurkar 2019)", "Reference"),
    "densenet121":         ("DenseNet-121 + GMP+GAP (ours)",                          "Ours"),
    "efficientnet_b4":     ("EfficientNet-B4 + GMP+GAP (ours)",                        "Ours"),
    "rad_dino":            ("Rad-DINO (ours)",                                         "Ours"),
}

if not os.path.exists(RESULTS):
    raise SystemExit(f"{RESULTS} not found. Download it from the server first.")

df = pd.read_csv(RESULTS)
df["Model"] = df["arch"].map(lambda a: META.get(a, (a, "Ours"))[0])
df["Role"]  = df["arch"].map(lambda a: META.get(a, (a, "Ours"))[1])

ref = df[df["Role"] == "Reference"]
if ref.empty:
    print("WARNING: no reference row yet (densenet121_vanilla). Run cv_reference_baseline.py")
    print("         on the server tonight, then re-download cv_results.csv.\n")
    ref_auc = None
else:
    ref_auc = float(ref["cv_mean_auc"].iloc[0])

df["delta_vs_ref"] = (df["cv_mean_auc"] - ref_auc) if ref_auc is not None else float("nan")

# order: your models by AUC (best first), reference last
ours = df[df["Role"] == "Ours"].sort_values("cv_mean_auc", ascending=False)
tab  = pd.concat([ours, ref], ignore_index=True)

# pretty columns
show = pd.DataFrame({
    "Model":          tab["Model"],
    "Role":           tab["Role"],
    "3-fold AUC":     tab["cv_mean_auc"].map(lambda x: f"{x:.4f}"),
    "+/- std":        tab["cv_std"].map(lambda x: f"{x:.4f}"),
    "fold1":          tab["fold1"].map(lambda x: f"{x:.4f}"),
    "fold2":          tab["fold2"].map(lambda x: f"{x:.4f}"),
    "fold3":          tab["fold3"].map(lambda x: f"{x:.4f}"),
    "delta vs ref":   tab["delta_vs_ref"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "-"),
})
show.to_csv(OUT, index=False)

print("=" * 78)
print("  CheXpert - 3-fold CV (90k images, patient-grouped, test set untouched)")
print("  Your models vs. the reference CheXpert-paper DenseNet-121")
print("=" * 78)
print(show.to_string(index=False))
print("=" * 78)
if ref_auc is not None:
    wins = ours[ours["cv_mean_auc"] > ref_auc]
    best = ours.iloc[0]
    print(f"Reference (vanilla DenseNet-121): {ref_auc:.4f}")
    print(f"Your best ({best['Model']}): {best['cv_mean_auc']:.4f} "
          f"({best['cv_mean_auc']-ref_auc:+.4f} over reference)")
    print(f"{len(wins)}/{len(ours)} of your models beat the reference on identical folds.")
print(f"\nsaved {OUT}  — this is the table to show Sir.")
