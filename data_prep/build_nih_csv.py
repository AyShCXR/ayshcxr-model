# build_nih_csv.py
# AyShCXR — builds a clean NIH ChestX-ray14 CSV (one-hot 14 labels + full image paths)
# from Data_Entry_2017.csv, matching the format train_model.py expects.
#
#   python build_nih_csv.py --root /workspace/kagglehub_cache
#   (point --root at wherever the NIH download was extracted)


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

import os, glob, argparse, pandas as pd

NIH_LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
              "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
              "Fibrosis","Pleural_Thickening","Hernia"]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace", help="folder where NIH was extracted")
ap.add_argument("--out",  default="/workspace/nih_clean.csv")
args = ap.parse_args()

# locate the label CSV
entry = next((p for p in glob.glob(os.path.join(args.root,"**","Data_Entry_2017*.csv"), recursive=True)), None)
assert entry, f"Data_Entry_2017.csv not found under {args.root}"
print("labels file:", entry)

# index every image file by filename (NIH spreads them across images_001..012/images/)
paths = {os.path.basename(p): p for p in glob.glob(os.path.join(args.root,"**","*.png"), recursive=True)}
print(f"indexed {len(paths):,} png files")

df = pd.read_csv(entry)
df["full_path"] = df["Image Index"].map(paths)
missing = int(df["full_path"].isna().sum())
print(f"rows {len(df):,} | unmatched images {missing:,}")
df = df.dropna(subset=["full_path"]).reset_index(drop=True)

for lab in NIH_LABELS:
    df[lab] = df["Finding Labels"].str.contains(lab, regex=False).astype("float32")
df["patient_id"] = df["Patient ID"].astype(str)

out_cols = ["full_path","patient_id"] + NIH_LABELS
df[out_cols].to_csv(args.out, index=False)
print(f"\nwrote {args.out} | {len(df):,} rows")
print("positives per label:")
print(df[NIH_LABELS].sum().astype(int).to_string())
