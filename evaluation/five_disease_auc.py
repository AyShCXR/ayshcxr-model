# five_disease_auc.py — per-disease AUC + the 5-disease subset, from saved preds.
# Uses val_predictions_densenet.csv (NIH model). Read-only.

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("val_predictions_densenet.csv")

# all 14 NIH disease columns
diseases = [c[len("true_"):] for c in df.columns if c.startswith("true_")]

# the 5 CheXpert competition diseases (Effusion == Pleural Effusion)
FIVE = ["Atelectasis", "Cardiomegaly", "Effusion", "Consolidation", "Edema"]

rows = []
for d in diseases:
    y = df[f"true_{d}"].values
    p = df[f"score_{d}"].values
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y, p)
    rows.append((d, auc, int(y.sum()), len(y)))

print(f"\nNIH DenseNet val predictions  ({len(df):,} images)\n")
print(f"{'Disease':<22}{'AUC':>8}{'#pos':>9}   (sorted)")
print("-" * 50)
for d, auc, pos, n in sorted(rows, key=lambda r: -r[1]):
    star = "  <-- in CheXpert-5" if d in FIVE else ""
    print(f"{d:<22}{auc:>8.4f}{pos:>9,}{star}")

all14 = np.nanmean([a for _, a, _, _ in rows])
five  = np.nanmean([a for d, a, _, _ in rows if d in FIVE])
print("-" * 50)
print(f"{'MEAN (all 14)':<22}{all14:>8.4f}")
print(f"{'MEAN (CheXpert-5)':<22}{five:>8.4f}")
print(f"\nsubset effect on NIH: {five - all14:+.4f}")
print()
