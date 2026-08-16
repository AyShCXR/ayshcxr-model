# fix_nih_paths.py
# Your nih_full_labels.csv already has all 14 labels — this ONLY re-points the
# 'full_path' column from Windows-relative paths to the real images on the server.
# Labels are NOT touched.
#
#   python fix_nih_paths.py

import os, glob, pandas as pd

ROOT = "/workspace/kagglehub_cache/datasets/nih-chest-xrays/data/versions/3"
CSV  = "/workspace/nih_full_labels.csv"

df = pd.read_csv(CSV)
print(f"loaded {CSV} | {len(df):,} rows | columns OK: {'Image Index' in df.columns}")

# match each Image Index to its actual file on the server (robust to folder layout)
paths = {os.path.basename(p): p for p in glob.glob(os.path.join(ROOT, "**", "*.png"), recursive=True)}
print(f"indexed {len(paths):,} png files under {ROOT}")

df["full_path"] = df["Image Index"].map(paths)
miss = int(df["full_path"].isna().sum())
print(f"unmatched images: {miss:,}")
df = df.dropna(subset=["full_path"]).reset_index(drop=True)

df.to_csv(CSV, index=False)
print(f"✅ fixed paths -> {CSV} | {len(df):,} usable rows")
print("example:", df['full_path'].iloc[0])
