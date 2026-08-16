# build_vindr_labels.py
# AyShCXR — VinDr-CXR box annotations -> image-level labels. 2026-08-10
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS EXISTS
#   VinDr-CXR is the only dataset here whose labels were written by RADIOLOGISTS
#   rather than by software. CheXpert's CheXBERT labels agree with radiologists
#   at only F1 ~0.44, and our own experiments showed three very different
#   architectures all plateauing at ~0.82 because of it. VinDr is the clean
#   reference that makes fine-tuning and honest evaluation possible.
#
# WHAT IT DOES
#   train.csv holds ONE ROW PER DRAWN BOX. Each image was read independently by
#   3 of 17 radiologists, and they frequently disagree. This collapses that to
#   one row per image with binary labels, using a configurable agreement rule.
#
# THE AGREEMENT RULE (--min-readers, default 2)
#   A finding counts as present only if at least N of the 3 readers marked it.
#     1 of 3 = sensitive, noisier    (any reader's suspicion counts)
#     2 of 3 = balanced  <- default, closest to how CheXpert's test set was built
#     3 of 3 = specific, very sparse (unanimous only)
#   The choice is recorded in the output so a later reader of your paper can see
#   exactly what "positive" meant.
#
# Output: data/vindr/vindr_labels.csv  — image_id, full_path, <canonical labels>,
#         n_readers, and a companion vindr_label_stats.csv
#
#   python data_prep/build_vindr_labels.py
#   python data_prep/build_vindr_labels.py --min-readers 1 --out other.csv

# ── AyShCXR path bootstrap ──────────────────────────────────────────────────
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(f"Cannot find .ayshcxr_root above {_HERE}")
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ────────────────────────────────────────────────────────────────────────────

import argparse
import pandas as pd

VINDR_DIR = _Path("data/vindr")
TRAIN_CSV = VINDR_DIR / "train.csv"
IMG_DIR   = VINDR_DIR / "train"

# ── VinDr class -> AyShCXR canonical name ───────────────────────────────────
# Mapped onto disease_ontology.CANONICAL_ORDER wherever a genuine equivalent
# exists. Entries mapped to None are deliberately dropped, with the reason
# stated — silently discarding data is how a dataset quietly misleads you.
VINDR_TO_CANONICAL = {
    "Atelectasis":         "Atelectasis",
    "Cardiomegaly":        "Cardiomegaly",
    "Consolidation":       "Consolidation",
    "Infiltration":        "Infiltration",
    "Lung Opacity":        "Lung Opacity",
    "Pleural effusion":    "Effusion",
    "Pleural thickening":  "Pleural Thickening",
    "Pneumothorax":        "Pneumothorax",
    "Pulmonary fibrosis":  "Fibrosis",
    "No finding":          "No Finding",

    # VinDr merges what NIH separates. Mapping this to either Nodule or Mass
    # alone would be wrong, so it becomes its own canonical finding —
    # "Lung Lesion" already exists in the ontology and means exactly this.
    "Nodule/Mass":         "Lung Lesion",

    # Genuinely new findings, not in the 21-canonical set yet. Kept as columns
    # so nothing is lost, but they cannot be used for fine-tuning until they
    # are added to disease_ontology.py and medical_knowledge_ext.py.
    "Aortic enlargement":  "Aortic Enlargement",
    "Calcification":       "Calcification",
    "ILD":                 "ILD",

    # Dropped: "Other lesion" is a catch-all with no consistent clinical
    # meaning. Training on it would teach the model to predict "something is
    # unusual", which is not a diagnosis and not actionable in a PHC.
    "Other lesion":        None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-readers", type=int, default=2,
                    help="readers who must agree for a finding to count (default 2 of 3)")
    ap.add_argument("--out", default="data/vindr/vindr_labels.csv")
    args = ap.parse_args()

    if not TRAIN_CSV.exists():
        _sys.exit(f"FATAL: {TRAIN_CSV} not found. Download train.csv from the "
                  f"VinBigData competition Data tab.")

    df = pd.read_csv(TRAIN_CSV)
    print("=" * 70)
    print("  VinDr-CXR — box annotations -> image-level labels")
    print("=" * 70)
    print(f"\nrows            : {len(df):,}")
    print(f"unique images   : {df.image_id.nunique():,}")
    print(f"radiologists    : {df.rad_id.nunique()}")
    print(f"agreement rule  : >= {args.min_readers} of 3 readers")

    # how many readers actually saw each image (should be 3)
    readers = df.groupby("image_id")["rad_id"].nunique()
    print(f"readers/image   : min {readers.min()}, max {readers.max()}, "
          f"mode {readers.mode().iloc[0]}")

    unknown = set(df.class_name.unique()) - set(VINDR_TO_CANONICAL)
    if unknown:
        print(f"\nWARNING: unmapped VinDr classes present: {unknown}")

    # ── count DISTINCT readers per (image, class) ───────────────────────────
    # Counting rows would be wrong: one radiologist may draw several boxes for
    # the same finding, which would inflate a single opinion into apparent
    # agreement.
    votes = (df.groupby(["image_id", "class_name"])["rad_id"]
               .nunique().reset_index(name="n_readers"))

    keep = {v: c for v, c in VINDR_TO_CANONICAL.items() if c}
    votes = votes[votes.class_name.isin(keep)].copy()
    votes["canonical"] = votes.class_name.map(keep)

    # Nodule/Mass and any future many-to-one merges: take the strongest vote
    votes = (votes.groupby(["image_id", "canonical"])["n_readers"]
                  .max().reset_index())

    votes["positive"] = (votes.n_readers >= args.min_readers).astype(int)

    wide = (votes.pivot(index="image_id", columns="canonical", values="positive")
                 .fillna(0).astype(int).reset_index())
    wide.columns.name = None

    # every image in train.csv, including any with no qualifying finding
    all_ids = pd.DataFrame({"image_id": sorted(df.image_id.unique())})
    wide = all_ids.merge(wide, on="image_id", how="left").fillna(0)
    label_cols = [c for c in wide.columns if c != "image_id"]
    wide[label_cols] = wide[label_cols].astype(int)

    # ── attach image paths, keep only rows whose PNG exists ─────────────────
    wide["full_path"] = wide.image_id.map(lambda i: str(IMG_DIR / f"{i}.png"))
    exists = wide.full_path.map(_os.path.exists)
    missing = int((~exists).sum())
    if missing:
        print(f"\nWARNING: {missing:,} images referenced but not on disk — dropped")
    wide = wide[exists].reset_index(drop=True)

    wide["n_readers"] = wide.image_id.map(readers).fillna(0).astype(int)
    ordered = ["image_id", "full_path", "n_readers"] + sorted(label_cols)
    wide = wide[ordered]

    out = _Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out, index=False)

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\nwrote {out}  ({len(wide):,} images x {len(label_cols)} findings)")

    try:
        from disease_ontology import CANONICAL_ORDER
        known = set(CANONICAL_ORDER)
    except ImportError:
        known = set()

    print(f"\n{'finding':<26}{'positives':>10}{'rate':>9}   status")
    print("-" * 62)
    stats = []
    for c in sorted(label_cols):
        n = int(wide[c].sum())
        rate = n / len(wide)
        if c not in known:
            status = "NEW — not in ontology yet"
        elif n < 300:
            status = "too sparse to train on"
        elif n < 1000:
            status = "usable, limited"
        else:
            status = "good"
        print(f"{c:<26}{n:>10,}{rate:>8.1%}   {status}")
        stats.append({"finding": c, "positives": n, "rate": round(rate, 4),
                      "in_ontology": c in known, "status": status})

    stats_path = out.with_name("vindr_label_stats.csv")
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    print(f"\nstats -> {stats_path}")

    trainable = [s for s in stats if s["in_ontology"] and s["positives"] >= 300
                 and s["finding"] != "No Finding"]
    print(f"\nfindings usable for fine-tuning right now: {len(trainable)}")
    for s in trainable:
        print(f"   {s['finding']:<26}{s['positives']:>7,}")

    new = [s["finding"] for s in stats if not s["in_ontology"]]
    if new:
        print(f"\nnot yet in the ontology (add to disease_ontology.py + "
              f"medical_knowledge_ext.py to use): {new}")

    print("\n" + "=" * 70)
    print("  DONE — dataset ready for fine-tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()
