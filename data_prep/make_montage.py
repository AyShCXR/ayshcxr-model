# make_montage.py
# Saves a grid of random RESIZED images + their ORIGINALS so we can SEE if the
# resize broke them. Open the two output PNGs in JupyterLab.

import os, glob, random
import numpy as np
from PIL import Image

RESIZED_ROOT  = "/chexpert_plus_png_412/PNG/train"
ORIGINAL_ROOT = "/chexpert_plus_png/PNG/train"

# collect 16 random resized images from DIFFERENT patients
patients = os.listdir(RESIZED_ROOT)
random.seed(1)
random.shuffle(patients)

picks = []
for pat in patients:
    pngs = glob.glob(os.path.join(RESIZED_ROOT, pat, "*", "*.png"))
    if pngs:
        picks.append(pngs[0])
    if len(picks) >= 16:
        break

def montage(paths, root_for_orig=None, cell=200):
    grid = Image.new("L", (cell*4, cell*4), 0)
    for i, p in enumerate(paths):
        src = p
        if root_for_orig:   # map resized path -> original path
            rel = os.path.relpath(p, RESIZED_ROOT)
            src = os.path.join(root_for_orig, rel)
        try:
            img = Image.open(src).convert("L").resize((cell, cell))
        except Exception as e:
            print(f"fail {src}: {e}"); continue
        arr = np.array(img)
        print(f"{os.path.relpath(p, RESIZED_ROOT):<55} mean={arr.mean():5.1f} std={arr.std():5.1f}")
        grid.paste(img, ((i % 4)*cell, (i // 4)*cell))
    return grid

print("=== RESIZED images (what the model actually trains on) ===")
g1 = montage(picks)
g1.save("/workspace/montage_resized.png")

print("\n=== ORIGINAL full-res images (same studies) ===")
g2 = montage(picks, root_for_orig=ORIGINAL_ROOT)
g2.save("/workspace/montage_original.png")

print("\nSaved /workspace/montage_resized.png and /workspace/montage_original.png")
print("Open BOTH in JupyterLab. They should look like 16 DIFFERENT, clear chest X-rays.")
print("If resized look scrambled/blank/identical but originals look fine -> resize is the bug.")
