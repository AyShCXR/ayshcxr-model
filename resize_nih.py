# resize_nih.py
# Resize all NIH images 1024 -> 412px (= IMG_SIZE 380 + 32, the training pipeline's
# FIRST Resize step), save grayscale PNGs to /workspace/nih_resized/, and repoint
# nih_full_labels.csv at the resized copies.
#
# Why this is safe: the model already downscales every image to 412px on the first
# transform, so it NEVER uses the 1024px detail. Pre-resizing gives identical training
# input, just ~3x faster epochs (same approach already used for CheXpert).
#
#   python resize_nih.py

import os, shutil, pandas as pd
from PIL import Image
from multiprocessing import Pool

CSV     = "/workspace/nih_full_labels.csv"
OUT_DIR = "/workspace/nih_resized"
SIZE    = 412                      # matches T.Resize((IMG_SIZE+32, IMG_SIZE+32))
os.makedirs(OUT_DIR, exist_ok=True)

try:                               # Pillow 10+ vs older
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

def resize_one(args):
    img_index, src = args
    dst = os.path.join(OUT_DIR, img_index)
    if os.path.exists(dst):
        return dst                 # idempotent — safe to re-run / resume
    try:
        Image.open(src).convert("L").resize((SIZE, SIZE), RESAMPLE).save(dst, "PNG")
        return dst
    except Exception as e:
        print("FAIL", src, e)
        return None

if __name__ == "__main__":
    df = pd.read_csv(CSV)
    if not os.path.exists(CSV + ".orig"):
        shutil.copy(CSV, CSV + ".orig")          # backup original paths (restorable)
        print(f"backed up original CSV -> {CSV}.orig")

    jobs = list(zip(df["Image Index"], df["full_path"]))
    print(f"resizing {len(jobs):,} images -> {OUT_DIR} at {SIZE}px (LANCZOS) ...")
    with Pool(8) as p:
        results = p.map(resize_one, jobs, chunksize=64)

    df["full_path"] = results
    failed = int(df["full_path"].isna().sum())
    df = df.dropna(subset=["full_path"]).reset_index(drop=True)
    df.to_csv(CSV, index=False)
    print(f"\n✅ done | {len(df):,} resized | {failed} failed | CSV repointed -> {CSV}")
    print("example:", df["full_path"].iloc[0])
