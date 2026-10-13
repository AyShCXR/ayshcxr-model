# site_calibrate_quality.py
# AyShCXR — re-derive image_quality thresholds for a new site, by the SAME rule.
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS EXISTS
#   image_quality.py thresholds sit at population percentiles of 400 VinDr-CXR
#   films (warn ~ p5/p95, severe ~ p1/p99). Run unchanged on NIH test films they
#   flag 39.6% as blurred (19.6% severe) and 25.8% as badly exposed, far above the
#   ~5% the rule is designed for. NIH images are 8-bit, 1024-px PNGs; Laplacian
#   variance and intensity statistics depend on that pipeline. The thresholds do
#   not transfer between sites.
#
#   The fix applies the identical percentile rule to UNLABELED films from the
#   new site's TRAINING split (never the test split). No label is used, nothing is
#   tuned to an outcome. This is also how a PHC deployment would work: score a few
#   hundred local films once, set thresholds, done.
#
# OUTPUT
#   results/quality_thresholds_nih.json  — {CONSTANT_NAME: value} overriding the
#   module constants of the same name in image_quality.py.

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import numpy as np
import pandas as pd

OUT = "results/quality_thresholds_nih.json"


def _raw(path):
    import image_quality as iq
    from PIL import Image
    try:
        a = iq._to_gray_array(Image.open(path))
    except Exception:
        return None
    r = iq.check_rotation(a)["value"]
    e = iq.check_exposure(a)["value"]
    s = iq.check_sharpness(a)["value"]
    return {"rotation": r, "mean": e["mean"], "std": e["std"],
            "dark_clip": e["dark_clip"], "bright_clip": e["bright_clip"], "sharpness": s}


def thresholds_from(v):
    """The documented rule from image_quality.py: warn ~ p5/p95, severe ~ p1/p99."""
    P = lambda k, q: float(np.percentile(v[k].dropna(), q))
    return {
        "ROTATION_WARN": P("rotation", 95), "ROTATION_SEVERE": P("rotation", 99),
        "DARK_MEAN": P("mean", 5), "DARK_MEAN_SEVERE": P("mean", 1),
        "BRIGHT_MEAN": P("mean", 95), "BRIGHT_MEAN_SEVERE": P("mean", 99),
        "LOW_CONTRAST_STD": P("std", 5),
        "BLUR_LAPLACIAN": P("sharpness", 5), "BLUR_SEVERE": P("sharpness", 1),
        "DARK_CLIP_WARN": P("dark_clip", 95), "BRIGHT_CLIP_WARN": P("bright_clip", 95),
    }


def apply(path=OUT):
    """Override image_quality's module constants in this process."""
    import image_quality as iq
    with open(path, encoding="utf-8") as f:
        t = json.load(f)["thresholds"]
    for k, v in t.items():
        setattr(iq, k, v)
    return t


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_csv("data/nih_full_labels.csv")
    with open("data/train_val_list.txt", encoding="utf-8") as f:
        tv = {ln.strip() for ln in f if ln.strip()}
    df = df[df["Image Index"].isin(tv)]
    df["rpath"] = df["full_path"].map(
        lambda p: str(p).lstrip(".").lstrip("\\/").replace("\\", "/"))
    df = df[df["rpath"].map(_os.path.exists)].sample(a.n, random_state=a.seed)

    with Pool(a.workers) as pool:
        vals = [v for v in pool.map(_raw, df["rpath"], chunksize=32) if v]
    v = pd.DataFrame(vals)

    import image_quality as iq
    new = thresholds_from(v)
    print(f"  NIH training films scored: {len(v)} (unlabeled; test split never touched)")
    print(f"  {'constant':20s} {'VinDr':>9s} {'NIH':>9s}")
    for k, x in new.items():
        print(f"  {k:20s} {getattr(iq, k):>9.4f} {x:>9.4f}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "NIH ChestX-ray14 train_val split, unlabeled",
                   "n_films": len(v), "seed": a.seed,
                   "rule": "warn ~ p5/p95, severe ~ p1/p99 (same as image_quality.py)",
                   "thresholds": new}, f, indent=2)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
