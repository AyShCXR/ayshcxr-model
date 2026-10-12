# label_ceiling.py
# AyShCXR — how much AUC do architectural improvements actually buy?
# by Subhrakant Sethi & Ayush Singh
#
# WHAT THIS REPORTS (and what it does NOT claim)
#   Reads results/cv_results.csv — a controlled 3-fold GroupKFold comparison on
#   IDENTICAL folds/recipe (see evaluation/cv_reference_baseline.py), where only
#   the architecture differs. It reports:
#     * each architecture's mean +/- std AUC
#     * the gain of each "ours" model over the CheXpert-paper reference
#       (vanilla DenseNet-121, Irvin & Rajpurkar 2019)
#     * the spread across architectures
#
#   HONEST FRAMING — this script deliberately does NOT assert "all architectures
#   converge". In the controlled CV they do NOT: they span a wide range. The
#   defensible, data-backed claim is narrower and is printed below:
#   complex head/pooling improvements on a fixed backbone buy very little over
#   the reference, i.e. diminishing returns from model complexity. The stronger
#   "labels, not capacity, are the ceiling" statement needs the label-agreement
#   measurement, which this script does NOT compute and must not be conflated
#   with the models' own F1.
#
# Self-test / run:  python reports/label_ceiling.py

import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = next((p for p in HERE.parents if (p / ".ayshcxr_root").exists()), None)
if ROOT:
    os.chdir(ROOT)
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CSV = "results/cv_results.csv"
REFERENCE = "densenet121_vanilla"      # CheXpert-paper style DenseNet (the baseline)

PRETTY = {
    "densenet121_vanilla": "DenseNet-121 (CheXpert paper reference)",
    "densenet121": "DenseNet-121 + GMP+GAP (ours)",
    "efficientnet_b4": "EfficientNet-B4 + GMP+GAP (ours)",
    "rad_dino": "Rad-DINO / ViT (ours)",
}


def main():
    if not os.path.exists(CSV):
        sys.exit(f"missing {CSV} — run evaluation/cv_reference_baseline.py first")
    df = pd.read_csv(CSV)

    if REFERENCE not in set(df["arch"]):
        sys.exit(f"reference row {REFERENCE!r} not in {CSV}; cannot compute gains")
    ref_auc = float(df.loc[df["arch"] == REFERENCE, "cv_mean_auc"].iloc[0])

    df = df.sort_values("cv_mean_auc", ascending=False).reset_index(drop=True)
    ours = df[df["arch"] != REFERENCE]
    spread = float(ours["cv_mean_auc"].max() - ours["cv_mean_auc"].min())
    best_gain = float(ours["cv_mean_auc"].max() - ref_auc)
    dn_gain = None
    if "densenet121" in set(df["arch"]):
        dn_gain = float(df.loc[df["arch"] == "densenet121", "cv_mean_auc"].iloc[0]
                        - ref_auc)

    print("=" * 78)
    print("  AyShCXR — architecture vs the label ceiling (controlled 3-fold CV)")
    print("=" * 78)
    print(f"\n  identical folds / recipe; only the architecture differs")
    print(f"  reference: {PRETTY.get(REFERENCE, REFERENCE)}  AUC {ref_auc:.4f}\n")
    print(f"  {'architecture':<42}{'CV AUC':>9}{'± std':>8}{'vs ref':>9}")
    print("  " + "-" * 66)
    for _, r in df.iterrows():
        delta = r["cv_mean_auc"] - ref_auc
        tag = "  (ref)" if r["arch"] == REFERENCE else f"{delta:+.4f}"
        print(f"  {PRETTY.get(r['arch'], r['arch']):<42}"
              f"{r['cv_mean_auc']:>9.4f}{r['cv_std']:>8.4f}{tag:>9}")
    print("  " + "-" * 66)

    print("\n  READING (data-backed, narrow):")
    print(f"    * Spread across architectures: {spread:.4f} AUC — these do NOT")
    print(f"      converge in the controlled CV, so 'architecture is irrelevant'")
    print(f"      is NOT supported and must not be claimed.")
    if dn_gain is not None:
        print(f"    * Our head/pooling changes on DenseNet gained {dn_gain:+.4f} over")
        print(f"      the reference DenseNet — a small return for real added complexity.")
    print(f"    * Best single architecture beats the reference by only {best_gain:+.4f}.")
    print("\n  DEFENSIBLE CLAIM FOR THE PAPER:")
    print("    Complex head/pooling improvements on a fixed backbone yield")
    print("    diminishing returns; the gains from model engineering are small.")
    print("    (The stronger 'labels are the ceiling' claim requires a separate")
    print("     NLP-label-vs-radiologist agreement measurement — NOT the model F1.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
