# resize_chexpert.py
# AyShCXR — one-time image pre-resize for fast, smooth training
# by Subhrakant Sethi & Ayush Singh
#
# Shrinks every full-resolution CheXpert PNG (~2300x2800, ~3MB) down to
# 412x412 grayscale (~30-50KB) ONCE on disk. After this:
#   • Each training batch reads ~100x less data → no disk fluctuation
#   • The whole resized set (~6-8GB) fits in the server's 1.4TB RAM cache
#   • Epoch 2 onward is served from memory → smooth + fast
#
# 412px chosen because the training pipeline does:
#   Resize(412,412) → RandomCrop(380)
# Pre-resizing to exactly 412x412 makes the Resize a no-op and keeps the crop working.
#
# Run once:  python resize_chexpert.py

import os
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

SRC_ROOT  = "/chexpert_plus_png/PNG"        # original full-res PNGs
DST_ROOT  = "/chexpert_plus_png_412/PNG"    # new resized output
RESIZE_TO = 412                              # matches Resize(IMG_SIZE+32) in training
NUM_PROC  = 64                               # parallel workers (server has 224 threads)

def find_all_pngs(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                paths.append(os.path.join(dirpath, fn))
    return paths

def resize_one(src_path):
    # mirror the directory structure under DST_ROOT
    rel      = os.path.relpath(src_path, SRC_ROOT)
    dst_path = os.path.join(DST_ROOT, rel)
    if os.path.exists(dst_path):
        return "skip"
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img = Image.open(src_path).convert("L")          # grayscale (training uses convert("L"))
        img = img.resize((RESIZE_TO, RESIZE_TO), Image.BILINEAR)
        img.save(dst_path, format="PNG", optimize=False)
        return "ok"
    except Exception as e:
        print(f"⚠️  Failed {src_path}: {e}")
        return "fail"

def main():
    print("Scanning source images...")
    all_pngs = find_all_pngs(SRC_ROOT)
    print(f"Found {len(all_pngs):,} PNG files")
    print(f"Resizing to {RESIZE_TO}x{RESIZE_TO} grayscale → {DST_ROOT}")
    print(f"Using {NUM_PROC} parallel workers\n")

    ok = skip = fail = 0
    with Pool(NUM_PROC) as pool:
        for result in tqdm(pool.imap_unordered(resize_one, all_pngs, chunksize=64),
                           total=len(all_pngs), desc="Resizing"):
            if   result == "ok":   ok   += 1
            elif result == "skip": skip += 1
            else:                  fail += 1

    print(f"\n✅ Done — resized: {ok:,} | skipped (already done): {skip:,} | failed: {fail:,}")
    print(f"   New image root: {DST_ROOT}")
    print(f"   Next: set PNG_ROOT = \"{DST_ROOT}\" in train_rad_dino_chexpert.py and restart")

if __name__ == "__main__":
    main()
