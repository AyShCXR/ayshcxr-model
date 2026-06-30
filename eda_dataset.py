# eda_dataset.py — characterize what we actually have on disk
# Reads the two label tables and prints structure, size, class balance,
# multi-label cardinality, and view mix. Read-only. Safe to re-run.

import os
import pandas as pd
import numpy as np

def section(t):
    print("\n" + "=" * 68)
    print("  " + t)
    print("=" * 68)

# ---- CheXpert-Plus (the professor-track dataset) -------------------
CHX = "chexpert_clean.csv"
CHX_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
PATHOLOGIES = [c for c in CHX_LABELS if c != "No Finding"]

if os.path.exists(CHX):
    df = pd.read_csv(CHX)
    section(f"CheXpert-Plus  ({CHX})")
    print(f"rows (images)     : {len(df):,}")
    print(f"unique patients   : {df['patient_id'].nunique():,}")
    print(f"images / patient  : {len(df)/df['patient_id'].nunique():.2f}")

    # view mix from the filename
    p = df["path_to_image"].astype(str)
    n_frontal = p.str.contains("frontal").sum()
    n_lateral = p.str.contains("lateral").sum()
    print(f"frontal / lateral : {n_frontal:,} / {n_lateral:,}")

    # values present in the label columns (is it binary, or -1/0/1 uncertainty?)
    vals = pd.unique(df[CHX_LABELS].values.ravel())
    print(f"label values seen : {sorted([v for v in vals if pd.notna(v)])}")

    print("\nper-label positive prevalence:")
    rows = []
    for c in CHX_LABELS:
        pos = int((df[c] == 1).sum())
        rows.append((c, pos, 100 * pos / len(df)))
    for c, pos, pct in sorted(rows, key=lambda r: -r[2]):
        bar = "#" * int(pct / 2)
        print(f"  {c:<28} {pos:>7,}  {pct:5.1f}%  {bar}")

    # multi-label cardinality over the 13 pathologies (exclude No Finding)
    card = (df[PATHOLOGIES] == 1).sum(axis=1)
    print(f"\nfindings per image (of 13 pathologies):")
    print(f"  mean              : {card.mean():.2f}")
    print(f"  images with 0     : {(card == 0).sum():,}  ({100*(card==0).mean():.1f}%)")
    print(f"  images with >=3   : {(card >= 3).sum():,}  ({100*(card>=3).mean():.1f}%)")
    print(f"  'No Finding'==1   : {int((df['No Finding']==1).sum()):,}")
else:
    print(f"(skip) {CHX} not found")

# ---- NIH ChestX-ray14 ----------------------------------------------
NIH = "nih_full_labels.csv"
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural Thickening", "Hernia",
]
if os.path.exists(NIH):
    df = pd.read_csv(NIH)
    section(f"NIH ChestX-ray14  ({NIH})")
    print(f"rows (images)     : {len(df):,}")
    if "Patient ID" in df.columns:
        print(f"unique patients   : {df['Patient ID'].nunique():,}")
    card = (df[NIH_LABELS] == 1).sum(axis=1)
    print(f"images with 0 findings (No Finding): {(card==0).sum():,}  ({100*(card==0).mean():.1f}%)")
    print("\nper-label positive prevalence:")
    rows = []
    for c in NIH_LABELS:
        pos = int((df[c] == 1).sum())
        rows.append((c, pos, 100 * pos / len(df)))
    for c, pos, pct in sorted(rows, key=lambda r: -r[2]):
        bar = "#" * int(pct / 2)
        print(f"  {c:<22} {pos:>7,}  {pct:5.1f}%  {bar}")
else:
    print(f"(skip) {NIH} not found")

print()
