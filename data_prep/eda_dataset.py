# eda_dataset.py — characterize what we actually have on disk
# Reads the two label tables and prints structure, size, class balance,
# multi-label cardinality, and view mix. Read-only. Safe to re-run.


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
