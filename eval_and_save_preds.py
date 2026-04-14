# eval_and_save_preds.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi & Ayush Singh
# Supports: DenseNet-121 (main) + EfficientNet-B4 (fallback)
#
# v3 fixes:
#   ✅ TTA horizontal flip REMOVED — clinically invalid for chest X-rays
#      A flipped chest X-ray shows heart on RIGHT side (dextrocardia pattern)
#      Model trained on normal orientation — flip confuses cardiac diseases
#   ✅ Replaced flip with rotate(+3 degrees) — clinically valid
#   ✅ num_workers=0 — required on Windows (no fork support)
#   ✅ Explicit model loading — loads confirmed 0.8031 safe model first

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("   AyShCXR — EVALUATION SCRIPT v3")
print("   by Subhrakant Sethi & Ayush Singh")
print("   TTA: 4 passes — NO horizontal flip (clinically invalid)")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

BATCH_SIZE = 32
IMG_SIZE   = 224

diseases = [
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
]

# ── Base transform (no augmentation) ─────────────────
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# ── TTA transforms ────────────────────────────────────
# v3 FIX: Removed horizontal flip — clinically invalid.
# A horizontally flipped chest X-ray shows the heart on
# the RIGHT side of the image. This is dextrocardia —
# a rare abnormality. Our model was trained on normal
# X-rays where the heart is always on the LEFT.
# Flipping caused Cardiomegaly AUC to drop by 0.035.
#
# Replacement augmentations — all clinically valid:
#   Version 1: Original (no change)
#   Version 2: Rotate +3 degrees (minor positioning variation)
#   Version 3: Brightness +10% (different exposure setting)
#   Version 4: Brightness -10% (underexposed scanner)
def get_tta_tensors(img_pil):
    """
    Returns 4 augmented versions of the same image.
    All are normalised and ready for model inference.
    NO horizontal flip — clinically invalid for chest X-rays.
    """
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    img_gray = img_pil.convert("L")
    base_resized = img_gray.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    versions = []

    # Version 1: Original (no augmentation)
    versions.append(normalize(base_resized))

    # Version 2: Slight rotation +3 degrees
    # Simulates minor patient positioning variation
    # Clinically valid — chest X-rays are never perfectly straight
    rotated = TF.rotate(base_resized, 3)
    versions.append(normalize(rotated))

    # Version 3: Slight brightness increase (+10%)
    # Simulates different exposure level on scanner
    brighter = TF.adjust_brightness(base_resized, 1.1)
    versions.append(normalize(brighter))

    # Version 4: Slight contrast increase (+10%)
    # Simulates different contrast setting on scanner
    contrasted = TF.adjust_contrast(base_resized, 1.1)
    versions.append(normalize(contrasted))

    return versions  # list of 4 tensors, shape [1, 224, 224]


# ── Load data ─────────────────────────────────────────
print("\nStep 1 — Loading labels...")
labels_df = pd.read_csv("nih_full_labels.csv")
print(f"Total: {len(labels_df):,}")

print("\nStep 2 — Loading test split...")
with open("test_list.txt", "r") as f:
    test_images = [line.strip() for line in f.readlines()]
print(f"Test images: {len(test_images):,}")

print("\nStep 3 — Filtering...")
test_df = labels_df[
    labels_df["Image Index"].isin(test_images)
].reset_index(drop=True)
print(f"With labels: {len(test_df):,}")

print("\nStep 4 — Finding images...")

def find_image_path(filename):
    for folder in os.listdir("."):
        if folder.startswith("images_"):
            path = os.path.join(folder, "images", filename)
            if os.path.exists(path):
                return path
    return None

test_df["resolved_path"] = test_df["Image Index"].apply(find_image_path)
found   = test_df["resolved_path"].notna().sum()
missing = test_df["resolved_path"].isna().sum()
print(f"Found: {found:,}  Missing: {missing:,}")
test_df = test_df[test_df["resolved_path"].notna()].reset_index(drop=True)

# ── Dataset ───────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img_pil = Image.open(row["resolved_path"])
            tensor  = base_transform(img_pil.convert("L"))
        except Exception:
            tensor = torch.zeros(1, IMG_SIZE, IMG_SIZE)
            img_pil = None
        labels = torch.tensor(
            row[diseases].values.astype(float),
            dtype=torch.float32
        )
        return tensor, labels, row["Image Index"], row["resolved_path"]

# ── Load model ────────────────────────────────────────
print("\nStep 5 — Loading model...")

# Priority 1 — explicitly named safe model (0.8031 confirmed)
SAFE_MODEL = "densenet121_BEST_auc0.8031_ep19_SAFE.pth"
if os.path.exists(SAFE_MODEL):
    print(f"✅ Found confirmed best model: {SAFE_MODEL}")
    MODEL_PATH = SAFE_MODEL
    MODEL_TYPE = "densenet"
else:
    # Priority 2 — search for available models
    candidates = [
        ("densenet121_14class_best.pth",      "densenet"),
        ("efficientnet_b4_14class_best.pth",  "efficientnet"),
    ]
    available = []
    for path, mtype in candidates:
        if os.path.exists(path):
            available.append((path, mtype, os.path.getmtime(path)))

    if not available:
        print("❌ No trained model found!")
        print("   Expected: densenet121_BEST_auc0.8031_ep19_SAFE.pth")
        exit()

    available.sort(key=lambda x: x[2], reverse=True)
    MODEL_PATH, MODEL_TYPE, _ = available[0]
    print(f"⚠️  Safe model not found. Using: {MODEL_PATH}")

print(f"Model path : {MODEL_PATH}")
print(f"Model type : {MODEL_TYPE.upper()}")

if MODEL_TYPE == "efficientnet":
    model    = models.efficientnet_b4(pretrained=False)
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(1, old_conv.out_channels, old_conv.kernel_size,
                         old_conv.stride, old_conv.padding, bias=False)
    model.features[0][0] = new_conv
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
        nn.Linear(in_f, 512), nn.ReLU(),
        nn.Dropout(p=0.3),   nn.Linear(512, 14)
    )
else:
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
        nn.Linear(in_f, 512), nn.ReLU(),
        nn.Dropout(p=0.3),   nn.Linear(512, 14)
    )

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Loaded {MODEL_PATH}")
except RuntimeError:
    print("⚠️  Trying old 2-layer architecture...")
    if MODEL_TYPE == "densenet":
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1024, 14))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Loaded {MODEL_PATH} (old arch)")

model.eval()
model.to(device)

# ── Run predictions with TTA ──────────────────────────
print("\nStep 6 — Running predictions with TTA (4 passes per image)...")
print("         Augmentations: original, rotate+3, bright+10%, contrast+10%")
print("         NO horizontal flip — clinically invalid for chest X-rays")

dataset    = XRayDataset(test_df)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,        # MUST be 0 on Windows — no fork support
    pin_memory=(device.type == "cuda")
)

all_preds  = []
all_labels = []
all_names  = []
total      = len(dataloader)

for i, (base_tensor, labels, names, paths) in enumerate(dataloader):
    if i % 20 == 0:
        pct = int(i / total * 100)
        print(f"  {pct}% ({i+1}/{total} batches)")

    batch_preds = []

    # Version 1 — original (base_tensor already computed by dataset)
    with torch.no_grad():
        out  = model(base_tensor.to(device))
        prob = torch.sigmoid(out).cpu().numpy()
    batch_preds.append(prob)

    # Versions 2-4 — additional augmentations per image in batch
    for aug_idx in range(1, 4):
        aug_tensors = []
        for path in paths:
            try:
                img_pil    = Image.open(path)
                tta_list   = get_tta_tensors(img_pil)
                aug_tensors.append(tta_list[aug_idx])
            except Exception:
                aug_tensors.append(torch.zeros(1, IMG_SIZE, IMG_SIZE))

        aug_batch = torch.stack(aug_tensors).to(device)
        with torch.no_grad():
            out  = model(aug_batch)
            prob = torch.sigmoid(out).cpu().numpy()
        batch_preds.append(prob)

    # Average predictions across all 4 augmentations
    tta_mean = np.mean(batch_preds, axis=0)
    all_preds.append(tta_mean)
    all_labels.append(labels.numpy())
    all_names.extend(list(names))

all_preds  = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
print(f"✅ TTA complete — evaluated {len(all_names):,} images × 4 augmentations")

# ── Save results ──────────────────────────────────────
print("\nStep 7 — Saving...")

save_name = (
    "val_predictions_efficientnet.csv"
    if MODEL_TYPE == "efficientnet"
    else "val_predictions_densenet.csv"
)

results_df = pd.DataFrame(
    all_preds, columns=[f"score_{d}" for d in diseases]
)
results_df.insert(0, "image_name", all_names)
for i, d in enumerate(diseases):
    results_df[f"true_{d}"] = all_labels[:, i]

results_df.to_csv(save_name, index=False)
print(f"✅ Saved → {save_name}")

# ── AUC Report ────────────────────────────────────────
print()
print("=" * 65)
print(f"   AyShCXR — PERFORMANCE REPORT (with fixed TTA)")
print(f"   Model : {MODEL_TYPE.upper()}")
print(f"   File  : {MODEL_PATH}")
print(f"   TTA   : original + rotate+3 + bright+10% + contrast+10%")
print("=" * 65)

scores = []
for disease in diseases:
    y_true = results_df[f"true_{disease}"].values
    y_pred = results_df[f"score_{disease}"].values
    pos    = int(y_true.sum())

    if len(np.unique(y_true)) < 2:
        print(f"SKIP  {disease:<22} no positive cases")
        continue

    try:
        auc = roc_auc_score(y_true, y_pred)
        scores.append(auc)
        if auc >= 0.80:
            grade = "✅ GREAT"
        elif auc >= 0.75:
            grade = "✅ GOOD "
        elif auc >= 0.60:
            grade = "⚠️  FAIR "
        else:
            grade = "❌ POOR "
        bar = "█" * int(auc * 20)
        print(f"{grade} {disease:<22} AUC={auc:.4f} pos={pos:,} {bar}")
    except Exception as e:
        print(f"SKIP  {disease:<22} {e}")

if scores:
    mean  = np.mean(scores)
    count = len(scores)
    print("-" * 65)
    print(f"  Mean AUC = {mean:.4f} (across {count} diseases, fixed TTA)")
    print()
    if mean >= 0.84:
        print("Overall → ✅ Excellent! Above 0.84 target")
    elif mean >= 0.82:
        print("Overall → ✅ Great — above 0.82")
    elif mean >= 0.80:
        print("Overall → ✅ Good — above radiologist benchmark")
    elif mean >= 0.79:
        print("Overall → ✅ On track — above radiologist 0.778")
    elif mean >= 0.60:
        print("Overall → ⚠️  Fair — needs more training")
    else:
        print("Overall → ❌ Needs more training")

    print()
    print(f"  Radiologist benchmark    : 0.778")
    print(f"  Your model (fixed TTA)   : {mean:.4f}")
    if mean > 0.778:
        print(f"  Above benchmark by       : +{mean-0.778:.4f} ✅")
    else:
        print(f"  Below benchmark by       : {mean-0.778:.4f} ❌")

print()
print("=" * 65)
print(f"Done! Results → {save_name}")
print("=" * 65)