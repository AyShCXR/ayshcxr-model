# eval_chexpert.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi & Ayush Singh
#
# Evaluates a CheXpert-trained model on the CheXpert Plus validation split.
# Matches the architecture and label mapping of build_and_train_chexpert.py exactly.
#
# Reports:
#   (1) Official CheXpert 5-disease benchmark AUC
#       (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion)
#   (2) Full 14-disease AUC (excluding soft-label-only diseases)
#
# Usage:
#   python eval_chexpert.py
#   python eval_chexpert.py --model chexpert_densenet_ep12_auc0.8500.pth
#
# ── BEFORE RUNNING: set CHEXPERT_ROOT below ────────────────────────────────
# Must match the CHEXPERT_ROOT you used in build_and_train_chexpert.py.
# ──────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

print("=" * 65)
print("   AyShCXR — CHEXPERT EVAL SCRIPT")
print("   by Subhrakant Sethi & Ayush Singh")
print("   TTA: original + rotate+3 + bright+10% + contrast+10%")
print("   NO horizontal flip — clinically invalid")
print("=" * 65)

# ── CONFIG — must match build_and_train_chexpert.py ───────────────────────
CHEXPERT_ROOT  = "/data/chexpert_plus"      # ← CHANGE THIS to actual path
CHEXPERT_CSV   = "df_chexpert_plus_240401.csv"
VAL_SPLIT      = 0.05
RANDOM_SEED    = 42
BATCH_SIZE     = 64
IMG_SIZE       = 380    # EfficientNet-B4 native resolution (must match training)
NUM_WORKERS    = 8      # set to 0 on Windows

# ── Device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU — will be slow")

# ── Label definitions ──────────────────────────────────────────────────────
TARGET_LABELS = (
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
)

# Official CheXpert 5-disease benchmark (Pleural Effusion = Effusion in our labels)
CHEXPERT_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

# Diseases with only soft labels in CheXpert — excluded from AUC (never hard 0/1)
SOFT_ONLY = {"Emphysema", "Fibrosis", "Pleural Thickening", "Hernia",
             "Infiltration", "Mass", "Nodule"}

MISSING_IN_CHEXPERT = {"Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"}

CHEXPERT_COLUMN_VARIANTS = {
    "Atelectasis"     : ["Atelectasis"],
    "Cardiomegaly"    : ["Cardiomegaly"],
    "Pleural Effusion": ["Pleural Effusion", "PleuralEffusion", "Pleural_Effusion"],
    "Consolidation"   : ["Consolidation"],
    "Edema"           : ["Edema"],
    "Pneumonia"       : ["Pneumonia"],
    "Pneumothorax"    : ["Pneumothorax"],
    "Lung Opacity"    : ["Lung Opacity", "LungOpacity", "Lung_Opacity"],
    "Lung Lesion"     : ["Lung Lesion", "LungLesion", "Lung_Lesion"],
}


def resolve_chexpert_cols(df_cols):
    resolved = {}
    df_cols_set = set(df_cols)
    for label_key, variants in CHEXPERT_COLUMN_VARIANTS.items():
        for v in variants:
            if v in df_cols_set:
                resolved[label_key] = v
                break
    return resolved


def map_chexpert_to_nih(row, resolved_cols):
    labels = {}

    def get_val(col_key):
        col = resolved_cols.get(col_key)
        if col is None:
            return None
        v = row.get(col, np.nan)
        if pd.isna(v):
            return 0.0
        v = float(v)
        return 0.5 if v == -1.0 else v

    labels["Atelectasis"]        = get_val("Atelectasis")     or 0.0
    labels["Cardiomegaly"]       = get_val("Cardiomegaly")    or 0.0
    labels["Effusion"]           = get_val("Pleural Effusion") or 0.0
    labels["Consolidation"]      = get_val("Consolidation")   or 0.0
    labels["Edema"]              = get_val("Edema")            or 0.0
    labels["Pneumonia"]          = get_val("Pneumonia")        or 0.0
    labels["Pneumothorax"]       = get_val("Pneumothorax")     or 0.0

    lo_val = get_val("Lung Opacity")
    ll_val = get_val("Lung Lesion")
    labels["Infiltration"]       = 0.5 if lo_val and lo_val > 0 else 0.0
    labels["Mass"]               = 0.5 if ll_val and ll_val > 0 else 0.0
    labels["Nodule"]             = 0.5 if ll_val and ll_val > 0 else 0.0

    for d in MISSING_IN_CHEXPERT:
        labels[d] = 0.5

    return labels


# ── Load and split data ────────────────────────────────────────────────────
print("\nStep 1 — Loading CheXpert Plus CSV...")
csv_path = os.path.join(CHEXPERT_ROOT, CHEXPERT_CSV)
if not os.path.exists(csv_path):
    print(f"❌ CSV not found: {csv_path}")
    print(f"   Set CHEXPERT_ROOT at the top of this file")
    sys.exit(1)

print(f"Reading {CHEXPERT_CSV} ... (405MB — may take ~30s)")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Total rows: {len(df):,}")

resolved_cols = resolve_chexpert_cols(df.columns)
if not resolved_cols:
    print("❌ Could not find CheXpert label columns in CSV!")
    sys.exit(1)

# Filter frontal only
view_col = next((c for c in ("Frontal/Lateral", "view", "View",
                              "frontal_lateral") if c in df.columns), None)
if view_col:
    before = len(df)
    df = df[df[view_col].str.lower() == "frontal"].reset_index(drop=True)
    print(f"Frontal only: {len(df):,} (was {before:,})")

print("\nStep 2 — Mapping labels to NIH 14-class format...")
label_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="  Mapping"):
    label_rows.append(map_chexpert_to_nih(row, resolved_cols))
label_df = pd.DataFrame(label_rows)
for col in TARGET_LABELS:
    df[col] = label_df[col].values
print("✅ Label mapping complete")

print("\nStep 3 — Getting val split...")
split_col = next((c for c in ("split", "Split", "partition") if c in df.columns), None)
if split_col:
    val_df = df[df[split_col].str.lower().isin(["valid", "val", "validation"])].reset_index(drop=True)
    print(f"Using built-in split → {len(val_df):,} val images")
else:
    pat_col = next((c for c in ("patient_id", "patient", "Patient") if c in df.columns), None)
    if pat_col:
        np.random.seed(RANDOM_SEED)
        patients = df[pat_col].unique()
        np.random.shuffle(patients)
        n_val = max(1, int(len(patients) * VAL_SPLIT))
        val_patients = set(patients[:n_val])
        val_df = df[df[pat_col].isin(val_patients)].reset_index(drop=True)
        print(f"Patient-level random split → {len(val_df):,} val images")
    else:
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
        val_df = val_df.reset_index(drop=True)
        print(f"Random image split → {len(val_df):,} val images")

path_col = next((c for c in ("Path", "path", "path_to_image") if c in df.columns), None)
if path_col and path_col != "Path":
    val_df = val_df.rename(columns={path_col: "Path"})

if len(val_df) == 0:
    print("❌ Val split is empty! Check split column values.")
    sys.exit(1)


# ── Dataset ────────────────────────────────────────────────────────────────
class CheXpertValDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        rel  = row.get("Path", "")
        path = os.path.join(CHEXPERT_ROOT, str(rel))
        try:
            img_pil = Image.open(path).convert("L")
            base    = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        except Exception:
            img_pil = None
            base    = Image.new("L", (IMG_SIZE, IMG_SIZE), 128)

        labels = torch.tensor(
            row[list(TARGET_LABELS)].to_numpy(dtype="float32")
        )
        return base, labels, str(rel), path


# ── TTA helper ────────────────────────────────────────────────────────────
normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229])
])

def get_tta_tensors(base_pil):
    """4 TTA versions — NO horizontal flip (clinically invalid for chest X-rays)."""
    return [
        normalize(base_pil),
        normalize(TF.rotate(base_pil, 3)),
        normalize(TF.adjust_brightness(base_pil, 1.1)),
        normalize(TF.adjust_contrast(base_pil, 1.1)),
    ]


# ── Model ─────────────────────────────────────────────────────────────────
print("\nStep 4 — Loading model...")

_model_override = next(
    (sys.argv[i+1] for i, a in enumerate(sys.argv[:-1]) if a == "--model"), None
)

def _find_chexpert_checkpoint():
    candidates = sorted(
        [f for f in os.listdir(".") if f.startswith("chexpert_efficientnet") and f.endswith(".pth")],
        key=lambda f: os.path.getmtime(f),
        reverse=True
    )
    return candidates[0] if candidates else None

if _model_override:
    MODEL_PATH = _model_override
    print(f"✅ Using override: {MODEL_PATH}")
else:
    MODEL_PATH = _find_chexpert_checkpoint()
    if MODEL_PATH is None:
        print("❌ No chexpert_densenet_*.pth checkpoint found.")
        print("   Run build_and_train_chexpert.py first, or pass --model <path>")
        sys.exit(1)
    print(f"✅ Found: {MODEL_PATH}")

# Build EfficientNet-B4 with GELU head (matches build_and_train_chexpert.py)
model = models.efficientnet_b4(weights=None)
old_conv = model.features[0][0]
new_conv = nn.Conv2d(1, 48,
                     kernel_size=old_conv.kernel_size,
                     stride=old_conv.stride,
                     padding=old_conv.padding,
                     bias=False)
model.features[0][0] = new_conv
in_f = model.classifier[1].in_features  # 1792
model.classifier = nn.Sequential(
    nn.BatchNorm1d(in_f),
    nn.Dropout(p=0.4),
    nn.Linear(in_f, 512),
    nn.GELU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 14)
)

def _load_weights(path):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    return ckpt

try:
    model.load_state_dict(_load_weights(MODEL_PATH))
    print(f"✅ Weights loaded — {MODEL_PATH} (EfficientNet-B4)")
except RuntimeError as e:
    print(f"❌ Failed to load weights: {e}")
    sys.exit(1)

model.eval()
model.to(device)


# ── Run predictions with TTA ───────────────────────────────────────────────
print(f"\nStep 5 — Running predictions with TTA on {len(val_df):,} images...")

dataset    = CheXpertValDataset(val_df)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    collate_fn=lambda batch: batch   # return list — we process images individually for TTA
)

all_preds  = []
all_labels = []
all_names  = []

# Process one batch at a time; TTA requires re-opening images
for batch in tqdm(dataloader, desc="  Evaluating"):
    batch_preds_per_aug = [[] for _ in range(4)]
    batch_labels = []
    batch_names  = []

    for base_pil, labels, rel_path, full_path in batch:
        batch_labels.append(labels.numpy())
        batch_names.append(rel_path)

        try:
            tta_tensors = get_tta_tensors(base_pil)
        except Exception:
            blank = torch.zeros(1, IMG_SIZE, IMG_SIZE)
            tta_tensors = [blank] * 4

        for aug_idx, t in enumerate(tta_tensors):
            batch_preds_per_aug[aug_idx].append(t)

    aug_means = []
    for aug_idx in range(4):
        aug_batch = torch.stack(batch_preds_per_aug[aug_idx]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(aug_batch)).cpu().numpy()
        aug_means.append(probs)

    tta_mean = np.mean(aug_means, axis=0)   # shape [batch, 14]
    all_preds.append(tta_mean)
    all_labels.append(np.stack(batch_labels))
    all_names.extend(batch_names)

all_preds  = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
print(f"✅ Done — {len(all_names):,} images × 4 TTA passes")


# ── Save predictions ───────────────────────────────────────────────────────
print("\nStep 6 — Saving predictions...")
results_df = pd.DataFrame(all_preds, columns=[f"score_{d}" for d in TARGET_LABELS])
results_df.insert(0, "path", all_names)
for i, d in enumerate(TARGET_LABELS):
    results_df[f"true_{d}"] = all_labels[:, i]
results_df.to_csv("val_predictions_chexpert.csv", index=False)
print("✅ Saved → val_predictions_chexpert.csv")


# ── AUC Report ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("   AyShCXR — CHEXPERT PERFORMANCE REPORT")
print(f"   Model  : {MODEL_PATH}")
print(f"   Images : {len(all_names):,}  |  TTA: 4 passes")
print("=" * 65)


def compute_auc(disease_list, label=""):
    scores = []
    skipped = []
    for disease in disease_list:
        y_true = results_df[f"true_{disease}"].values
        y_pred = results_df[f"score_{disease}"].values
        hard_pos = (y_true == 1.0).sum()
        hard_neg = (y_true == 0.0).sum()
        if hard_pos < 5 or hard_neg < 5:
            skipped.append(disease)
            continue
        try:
            auc = roc_auc_score(y_true, y_pred)
            scores.append((disease, auc, int(hard_pos)))
        except Exception as e:
            skipped.append(f"{disease}({e})")

    if label:
        print(f"\n── {label} ──")

    for disease, auc, pos in scores:
        if auc >= 0.80:
            grade = "✅ GREAT"
        elif auc >= 0.75:
            grade = "✅ GOOD "
        elif auc >= 0.60:
            grade = "⚠️  FAIR "
        else:
            grade = "❌ POOR "
        bar = "█" * int(auc * 20)
        print(f"{grade} {disease:<22} AUC={auc:.4f}  pos={pos:,}  {bar}")

    if skipped:
        print(f"  SKIPPED (no hard labels): {', '.join(skipped)}")

    if scores:
        mean = np.mean([s[1] for s in scores])
        print(f"  {'Mean AUC':<24} {mean:.4f}  (n={len(scores)})")
        return mean, scores
    return None, []


# ① Official CheXpert 5-disease benchmark
mean5, scores5 = compute_auc(
    CHEXPERT_5,
    label="OFFICIAL CHEXPERT 5-DISEASE BENCHMARK"
)

# ② Full 14-disease (excluding soft-label-only diseases)
hard_diseases = [d for d in TARGET_LABELS if d not in SOFT_ONLY]
mean14, scores14 = compute_auc(
    hard_diseases,
    label="FULL 14-DISEASE AUC (hard-label diseases only)"
)

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("   SUMMARY")
print("=" * 65)

if mean5 is not None:
    print(f"\n  CheXpert 5-disease AUC : {mean5:.4f}")
    print(f"  CheXpert leaderboard   : 0.930 (state of art, ensemble)")
    print(f"  CheXpert paper baseline: 0.850 (DenseNet-121 original)")
    if mean5 >= 0.90:
        print("  Overall → ✅ Excellent! Above 0.90")
    elif mean5 >= 0.87:
        print("  Overall → ✅ Great — approaching ensemble level")
    elif mean5 >= 0.85:
        print("  Overall → ✅ Good — matches original CheXpert paper")
    elif mean5 >= 0.80:
        print("  Overall → ✅ Above 0.80 — solid result")
    else:
        print("  Overall → ⚠️  Needs more training")

if mean14 is not None:
    print(f"\n  14-disease AUC (hard)  : {mean14:.4f}")
    print(f"  NIH radiologist bench  : 0.778")
    print(f"  AyShCXR NIH test best  : 0.8031")
    if mean14 > 0.8031:
        print(f"  Above NIH best by      : +{mean14-0.8031:.4f} ✅")
    else:
        print(f"  Below NIH best by      : {mean14-0.8031:.4f}")

print()
print("=" * 65)
print("Done! Results → val_predictions_chexpert.csv")
print("=" * 65)
