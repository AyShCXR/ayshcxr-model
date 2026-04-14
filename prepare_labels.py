# prepare_labels.py
# Prepares labels for ALL 14 diseases from NIH dataset
# Reads Data_Entry_2017.csv and creates nih_full_labels.csv

import pandas as pd
import os

print("=" * 55)
print("   LABEL PREPARATION SCRIPT")
print("   Expanding from 4 to 14 diseases")
print("=" * 55)

# ── All 14 NIH disease labels ─────────────────────────
TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia"
]

# ── Step 1 — Load CSV ─────────────────────────────────
print("\nStep 1 — Loading Data_Entry_2017.csv...")
df = pd.read_csv("Data_Entry_2017.csv")
print(f"Total rows in CSV : {len(df)}")
print(f"Columns           : {df.columns.tolist()}")

# ── Step 2 — Create binary columns for all 14 ────────
print("\nStep 2 — Creating binary labels for 14 diseases...")
NIH_NAME_MAP = {"Pleural Thickening": "Pleural_Thickening"}

for disease in TARGET_LABELS:
    nih_name = NIH_NAME_MAP.get(disease, disease)
    df[disease] = df["Finding Labels"].fillna("").apply(
        lambda x: 1 if nih_name in x.split("|") else 0
    )

# ── Step 3 — Show class distribution ─────────────────
print("\nStep 3 — Class distribution:")
print(f"{'Disease':<20} {'Positive':>10} {'Negative':>10} {'%':>8}")
print("-" * 52)
for disease in TARGET_LABELS:
    pos = df[disease].sum()
    neg = len(df) - pos
    pct = pos / len(df) * 100
    print(f"{disease:<20} {pos:>10,} {neg:>10,} {pct:>7.2f}%")

# ── Step 4 — Build image path mapping ────────────────
print("\nStep 4 — Finding image files on disk...")

def build_path_map(root_dir="."):
    mapping = {}
    for folder in os.listdir(root_dir):
        if not folder.startswith("images_"):
            continue
        for dirpath, _, filenames in os.walk(
            os.path.join(root_dir, folder)
        ):
            for fname in filenames:
                if fname.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    mapping[fname] = os.path.join(
                        dirpath, fname
                    )
    return mapping

path_map = build_path_map()
print(f"Total images found on disk: {len(path_map):,}")

# ── Step 5 — Add full path column ────────────────────
print("\nStep 5 — Matching image names to paths...")
df["full_path"] = df["Image Index"].map(path_map)

before = len(df)
df = df[df["full_path"].notna()].reset_index(drop=True)
after = len(df)
print(f"Images matched  : {after:,}")
print(f"Images missing  : {before - after:,}")

# ── Step 6 — Save full labels CSV ────────────────────
print("\nStep 6 — Saving nih_full_labels.csv...")

save_cols = ["Image Index", "Patient ID", "full_path"] + TARGET_LABELS
df[save_cols].to_csv("nih_full_labels.csv", index=False)

print(f"✅ nih_full_labels.csv saved!")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {len(save_cols)}")
print(f"   Diseases: {len(TARGET_LABELS)}")

# ── Step 7 — Also save the old name for compatibility ─
df[save_cols].to_csv("nih_demo_labels.csv", index=False)
print(f"✅ nih_demo_labels.csv also updated (compatibility)")

print()
print("=" * 55)
print("Label preparation complete!")
print("Next step: run build_and_train_demo.py")
print("=" * 55)