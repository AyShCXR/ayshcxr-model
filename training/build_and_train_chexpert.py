# build_and_train_chexpert.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi & Ayush Singh
#
# ── CHEXPERT PLUS — FRESH GPU RUN ─────────────────────────────────────────
# Model    : EfficientNet-B4 at 380px  ← stronger backbone for paper headline
# Dataset  : CheXpert Plus (223,462 images + 187,711 radiology reports)
# Purpose  : CheXpert Plus training, val on CheXpert val split
# GPU      : Professor's A100 (80GB VRAM)
# Baseline : 0.8031 (DenseNet-121 on NIH test) — aim to beat after NIH test eval
#
# CheXpert Plus label mapping:
#   DIRECT (7): Atelectasis, Cardiomegaly, Effusion←PleuralEffusion,
#               Consolidation, Edema, Pneumonia, Pneumothorax
#   SOFT 0.5 (3 uncertain): Infiltration←LungOpacity,
#                           Mass←LungLesion, Nodule←LungLesion
#   MISSING → 0.5 (4): Emphysema, Fibrosis, Pleural Thickening, Hernia
#   UNCERTAIN LABELS (-1 in CSV) → 0.5
#   NaN → 0.0 (not annotated)
#
# INDICATION text: extracted from CSV and saved for later symptom fusion
#   File: chexpert_indication_texts.csv
#   Format: patient_id, study_id, path, indication_text
#
# Checkpoint prefix: chexpert_efficientnet_* (no conflict with other .pth files)
# DO NOT MODIFY: build_and_train_demo.py (EfficientNet-B4 NIH run)
# DO NOT MODIFY: build_and_train_laptop.py (DenseNet-121 NIH laptop run)
#
# Usage on professor's GPU:
#   python build_and_train_chexpert.py
#
# ─── BEFORE RUNNING: set CHEXPERT_ROOT below ───────────────────────────────
# Point CHEXPERT_ROOT to the folder that CONTAINS df_chexpert_plus_240401.csv
# and the image PNG folder (usually CheXpert-v1.0/ or a similar subfolder).
#
#   Example: CHEXPERT_ROOT = "/data/chexpert_plus"
#
#   The images must be reachable at: os.path.join(CHEXPERT_ROOT, row["Path"])
#   where row["Path"] looks like "CheXpert-v1.0/train/patient00001/..."
# ──────────────────────────────────────────────────────────────────────────

import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── CONFIG — set CHEXPERT_ROOT before running ─────────────────────────────
CHEXPERT_ROOT = "/data/chexpert_plus"   # ← CHANGE THIS to actual path on GPU

CHEXPERT_CSV  = "df_chexpert_plus_240401.csv"   # relative to CHEXPERT_ROOT

RANDOM_SEED        = 42
BATCH_SIZE         = 128   # auto-adjusted below based on VRAM
NUM_WORKERS        = 8
NUM_EPOCHS         = 70
IMG_SIZE           = 380   # EfficientNet-B4 native resolution
LR_HEAD            = 3e-5
LR_BACKBONE        = 1e-5
PATIENCE           = 12    # counts AUC epochs only (every 3 epochs = 36 raw epochs)
MIN_EPOCHS_SAVE    = 5     # save checkpoints from epoch 5 onwards
MIN_EPOCHS_STOP    = 35    # no early stopping before epoch 35
VAL_SPLIT          = 0.05  # use CheXpert built-in split if available
LABEL_SMOOTH       = 0.1
USE_AMP            = True
AUC_EVERY_N        = 3
CKPT_EVERY_N       = 5
AUTO_RESUME        = True
ACCUMULATION_STEPS = 1     # A100 80GB — no accumulation needed for batch=128

# ── NIH 14-class target labels (same order as all other AyShCXR models) ──
TARGET_LABELS = (
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
)

DISEASE_LOSS_WEIGHTS = {
    "Atelectasis"       : 0.57,
    "Cardiomegaly"      : 1.21,
    "Effusion"          : 0.53,
    "Infiltration"      : 0.50,
    "Mass"              : 0.83,
    "Nodule"            : 0.79,
    "Pneumonia"         : 2.00,
    "Pneumothorax"      : 0.87,
    "Consolidation"     : 0.93,
    "Edema"             : 1.33,
    "Emphysema"         : 1.27,   # soft label 0.5 — upweight so model doesn't ignore
    "Fibrosis"          : 1.56,   # soft label 0.5
    "Pleural Thickening": 0.88,
    "Hernia"            : 3.00,   # soft label 0.5
}

# ── CheXpert Plus label columns → NIH target mapping ─────────────────────
# Values in CheXpert Plus: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), NaN
#
# Possible column name variants across CheXpert releases:
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

# NIH targets that have NO corresponding CheXpert column → use soft 0.5
MISSING_IN_CHEXPERT = {"Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"}


def resolve_chexpert_cols(df_cols):
    """Find actual column names in the CSV for each CheXpert label."""
    resolved = {}
    df_cols_set = set(df_cols)
    for label_key, variants in CHEXPERT_COLUMN_VARIANTS.items():
        for v in variants:
            if v in df_cols_set:
                resolved[label_key] = v
                break
    return resolved


def map_chexpert_to_nih(row, resolved_cols):
    """Convert one CheXpert Plus row to NIH 14-class label vector.

    -1 (uncertain) → 0.5 (soft positive)
    NaN            → 0.0 (not annotated = assume negative)
    1              → 1.0 or 0.5 (soft) depending on mapping
    0              → 0.0
    """
    labels = {}

    def get_val(col_key):
        col = resolved_cols.get(col_key)
        if col is None:
            return None
        v = row.get(col, np.nan)
        if pd.isna(v):
            return 0.0
        v = float(v)
        if v == -1.0:
            return 0.5   # uncertain → soft positive
        return v         # 0.0 or 1.0

    # Direct mappings (7 diseases)
    labels["Atelectasis"]  = get_val("Atelectasis")  or 0.0
    labels["Cardiomegaly"] = get_val("Cardiomegaly") or 0.0
    labels["Effusion"]     = get_val("Pleural Effusion") or 0.0
    labels["Consolidation"]= get_val("Consolidation") or 0.0
    labels["Edema"]        = get_val("Edema")        or 0.0
    labels["Pneumonia"]    = get_val("Pneumonia")    or 0.0
    labels["Pneumothorax"] = get_val("Pneumothorax") or 0.0

    # Soft label mappings (3 diseases)
    lo_val  = get_val("Lung Opacity")
    ll_val  = get_val("Lung Lesion")

    # LungOpacity → Infiltration: if positive/uncertain → 0.5 (soft)
    labels["Infiltration"] = 0.5 if lo_val and lo_val > 0 else 0.0

    # LungLesion → both Mass and Nodule: if positive/uncertain → 0.5 each
    labels["Mass"]         = 0.5 if ll_val and ll_val > 0 else 0.0
    labels["Nodule"]       = 0.5 if ll_val and ll_val > 0 else 0.0

    # Missing diseases → soft 0.5 (model learns frequency pattern)
    for d in MISSING_IN_CHEXPERT:
        labels[d] = 0.5

    return labels


# ── Device ────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory
        print(f"✅ GPU: {name}")
        print(f"   VRAM: {vram/1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected — will be very slow")
    return device


# ── Dataset ───────────────────────────────────────────────────────────────
class CheXpertDataset(Dataset):
    def __init__(self, df, transform, chexpert_root):
        self.df            = df.reset_index(drop=True)
        self.transform     = transform
        self.chexpert_root = chexpert_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        rel  = row.get("Path") or row.get("path") or row.get("path_to_image") or ""
        path = os.path.join(self.chexpert_root, str(rel))

        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")
            img = Image.new("L", (IMG_SIZE, IMG_SIZE), 128)

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(
            row[list(TARGET_LABELS)].to_numpy(dtype="float32")
        )
        return img, labels


# ── Focal Loss ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75,
                 smoothing=0.1, disease_weights=None):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.smoothing       = smoothing
        self.disease_weights = disease_weights

    def forward(self, inputs, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(
                       inputs, targets_smooth, reduction='none')
        pt    = torch.exp(-bce_loss)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce_loss
        if self.disease_weights is not None:
            focal = focal * self.disease_weights.to(inputs.device)
        return focal.mean()


# ── EfficientNet-B4 ───────────────────────────────────────────────────────
def create_efficientnet(num_classes=14):
    print(f"Building EfficientNet-B4 at {IMG_SIZE}px (CheXpert Plus GPU run)...")
    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )

    # Grayscale input (1-channel): average RGB weights to preserve transfer
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        1, 48,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    model.features[0][0] = new_conv

    in_f = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Dropout(p=0.4),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"✅ EfficientNet-B4 ready — {total/1e6:.1f}M parameters")
    print(f"   Input resolution : {IMG_SIZE}px")
    print(f"   Feature channels : {in_f} → 512 → {num_classes}")
    return model


# ── Optimizer — layer-wise LR decay ───────────────────────────────────────
def get_optimizer(model):
    early_params, middle_params, late_params, head_params = [], [], [], []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        elif "features.0." in name or "features.1." in name:
            early_params.append(param)   # stem + first MBConv — barely touch
        elif "features.2." in name or "features.3." in name or "features.4." in name:
            middle_params.append(param)  # mid backbone — adapt slowly
        else:
            late_params.append(param)    # features.5-8 — adapt more freely

    return optim.AdamW([
        {"params": early_params,  "lr": 1e-6,        "weight_decay": 1e-5},
        {"params": middle_params, "lr": 5e-6,        "weight_decay": 1e-5},
        {"params": late_params,   "lr": LR_BACKBONE, "weight_decay": 1e-5},
        {"params": head_params,   "lr": LR_HEAD,     "weight_decay": 1e-4},
    ])


# ── Transforms ────────────────────────────────────────────────────────────
def get_train_transform():
    return T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomCrop(IMG_SIZE),
        # No horizontal flip — heart is left-sided
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229]),
        T.RandomErasing(p=0.1, value=0.485),
    ])


def get_val_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])


# ── AUC ───────────────────────────────────────────────────────────────────
def compute_val_auc(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            probs = torch.sigmoid(model(imgs.to(device)))
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    auc_dict = {}
    for i, disease in enumerate(TARGET_LABELS):
        y_true = all_labels[:, i]
        # Skip diseases with constant labels (e.g., Hernia=0.5 everywhere)
        unique_vals = np.unique(y_true)
        if len(unique_vals) < 2:
            continue
        # Need at least some hard-label positives for meaningful AUC
        hard_pos = (y_true == 1.0).sum()
        hard_neg = (y_true == 0.0).sum()
        if hard_pos < 5 or hard_neg < 5:
            continue
        try:
            auc_dict[disease] = roc_auc_score(y_true, all_preds[:, i])
        except Exception as e:
            print(f"⚠️  AUC failed for {disease}: {e}")
    mean_auc = float(np.mean(list(auc_dict.values()))) if auc_dict else 0.0
    return mean_auc, auc_dict


# ── INDICATION text extractor ─────────────────────────────────────────────
def extract_indication_texts(df, out_csv="chexpert_indication_texts.csv"):
    """Save raw INDICATION text per image for later symptom fusion training."""
    text_col = None
    for candidate in ("indication", "Indication", "INDICATION",
                       "report", "Report", "impression", "Impression",
                       "clinical_history", "Clinical History"):
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        print("⚠️  No INDICATION/report column found — skipping text extraction")
        print(f"   Available columns: {list(df.columns[:20])}")
        return

    path_col = next((c for c in ("Path", "path", "path_to_image") if c in df.columns), None)
    id_col   = next((c for c in ("patient_id", "patient", "Patient") if c in df.columns), None)

    rows = []
    for _, row in df.iterrows():
        text = str(row[text_col]) if not pd.isna(row.get(text_col, np.nan)) else ""
        if not text or text == "nan":
            continue
        rows.append({
            "patient_id"     : row.get(id_col, "") if id_col else "",
            "path"           : row.get(path_col, "") if path_col else "",
            "indication_text": text.strip(),
        })

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"✅ Saved {len(rows):,} indication texts → {out_csv}")
    else:
        print("⚠️  No non-empty indication texts found")


# ── Load CheXpert Plus ─────────────────────────────────────────────────────
def load_chexpert_data():
    print("\n── Loading CheXpert Plus ──────────────────────────────")
    csv_path = os.path.join(CHEXPERT_ROOT, CHEXPERT_CSV)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        print(f"   Set CHEXPERT_ROOT at the top of this file")
        return None, None, None

    print(f"Reading {CHEXPERT_CSV} ... (405MB — may take ~30s)")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total rows in CSV : {len(df):,}")
    print(f"Columns (first 30): {list(df.columns[:30])}")

    # ── Resolve actual column names for CheXpert labels ───────────────
    resolved_cols = resolve_chexpert_cols(df.columns)
    print(f"Resolved CheXpert label columns: {resolved_cols}")
    if not resolved_cols:
        print("❌ Could not find any CheXpert label columns!")
        print("   Check that df_chexpert_plus_240401.csv has disease columns.")
        return None, None, None

    # ── Filter to frontal view only (lateral rarely annotated) ────────
    view_col = next((c for c in ("Frontal/Lateral", "view", "View",
                                 "frontal_lateral") if c in df.columns), None)
    if view_col:
        before = len(df)
        df = df[df[view_col].str.lower() == "frontal"].reset_index(drop=True)
        print(f"Frontal only     : {len(df):,} (was {before:,})")

    # ── INDICATION text: save for later symptom fusion ────────────────
    if not os.path.exists("chexpert_indication_texts.csv"):
        print("Extracting INDICATION texts...")
        extract_indication_texts(df)
    else:
        print("chexpert_indication_texts.csv already exists — skipping extraction")

    # ── Map CheXpert labels → NIH 14-class ────────────────────────────
    print("Mapping CheXpert labels to NIH 14-class format...")
    label_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Mapping labels"):
        label_rows.append(map_chexpert_to_nih(row, resolved_cols))
    label_df = pd.DataFrame(label_rows)
    for col in TARGET_LABELS:
        df[col] = label_df[col].values
    print("✅ Label mapping complete")

    # ── Train/val split ───────────────────────────────────────────────
    split_col = next((c for c in ("split", "Split", "partition") if c in df.columns), None)
    if split_col:
        train_df = df[df[split_col].str.lower() == "train"].reset_index(drop=True)
        val_df   = df[df[split_col].str.lower().isin(["valid", "val", "validation"])].reset_index(drop=True)
        print(f"Using built-in split:")
    else:
        # No split column — do random split by patient if patient col exists
        pat_col = next((c for c in ("patient_id", "patient", "Patient") if c in df.columns), None)
        if pat_col:
            patients = df[pat_col].unique()
            np.random.shuffle(patients)
            n_val = max(1, int(len(patients) * VAL_SPLIT))
            val_patients = set(patients[:n_val])
            train_df = df[~df[pat_col].isin(val_patients)].reset_index(drop=True)
            val_df   = df[df[pat_col].isin(val_patients)].reset_index(drop=True)
            print(f"Patient-level random split (no built-in split column found):")
        else:
            # Fallback: random image split
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, test_size=VAL_SPLIT, random_state=RANDOM_SEED
            )
            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
            print(f"Random image split (no patient or split column found):")

    print(f"  Train : {len(train_df):,} images")
    print(f"  Val   : {len(val_df):,} images")

    # ── Sanity: confirm image path resolution ─────────────────────────
    path_col = next((c for c in ("Path", "path", "path_to_image") if c in df.columns), None)
    if path_col and len(train_df) > 0:
        sample_path = os.path.join(CHEXPERT_ROOT, str(train_df.iloc[0][path_col]))
        if not os.path.exists(sample_path):
            print(f"⚠️  WARNING: Sample image not found at: {sample_path}")
            print(f"   Check CHEXPERT_ROOT = '{CHEXPERT_ROOT}'")
            print(f"   And that {path_col} column contains relative paths")
        else:
            print(f"✅ Image path check passed: {sample_path}")

    # Ensure path col is called "Path" for the Dataset class
    if path_col and path_col != "Path":
        train_df = train_df.rename(columns={path_col: "Path"})
        val_df   = val_df.rename(columns={path_col: "Path"})

    return train_df, val_df, CHEXPERT_ROOT


# ── Training loop ─────────────────────────────────────────────────────────
def train_loop(model, train_df, val_loader, device, optimizer, scheduler,
               start_epoch=1, initial_best_auc=0.0):

    loss_weights = torch.tensor(
        [DISEASE_LOSS_WEIGHTS[d] for d in TARGET_LABELS], dtype=torch.float32
    )
    criterion = FocalLoss(
        gamma=2.0, alpha=0.75,
        smoothing=LABEL_SMOOTH,
        disease_weights=loss_weights
    )

    use_amp = USE_AMP and device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    best_auc        = initial_best_auc
    patience_count  = 0
    best_model_name = None
    history         = []
    prev_auc        = None

    train_loader = DataLoader(
        CheXpertDataset(train_df, get_train_transform(), CHEXPERT_ROOT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    print("\n" + "=" * 65)
    print("   AyShCXR — EFFICIENTNET-B4 CHEXPERT PLUS GPU RUN")
    print(f"   Training images  : {len(train_df):,}")
    print(f"   Val images       : {len(val_loader.dataset):,}")
    print(f"   Architecture     : EfficientNet-B4 (~19M parameters)")
    print(f"   Loss             : Focal Loss (gamma=2.0, alpha=0.75)")
    print(f"   Image size       : {IMG_SIZE}px")
    print(f"   Epochs           : {NUM_EPOCHS} max, patience={PATIENCE}")
    print(f"   Batch size       : {BATCH_SIZE}  (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"   AMP              : {'enabled' if use_amp else 'disabled'}")
    print(f"   Resuming from    : epoch {start_epoch}, best AUC {best_auc:.4f}")
    print(f"   NIH EfficientNet : 0.8031 DenseNet ref / aim higher with B4")
    print(f"   Goal             : CheXpert Plus baseline → later NIH test eval")
    print("=" * 65)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch:02d}/{NUM_EPOCHS}  |  {IMG_SIZE}px  |  {len(train_df):,} images")

        # ── Train ─────────────────────────────────────────
        model.train()
        train_loss, train_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc="  Train")):
            imgs, labels = imgs.to(device), labels.to(device)

            # Mixup — inter-patient variation
            lam  = np.random.beta(0.4, 0.4)
            idx  = torch.randperm(imgs.size(0), device=device)
            imgs   = lam * imgs   + (1 - lam) * imgs[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs), labels) / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                loss = criterion(model(imgs), labels) / ACCUMULATION_STEPS
                loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss  += loss.item() * ACCUMULATION_STEPS
            train_batches += 1

        avg_train = train_loss / train_batches
        scheduler.step()

        # ── Val loss (odd epochs only — informational) ────
        if epoch % 2 != 0:
            model.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    val_loss    += criterion(model(imgs), labels).item()
                    val_batches += 1
            avg_val = val_loss / val_batches
        else:
            avg_val = 0.0

        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"  Train Loss : {avg_train:.4f}")
        if avg_val > 0:
            print(f"  Val Loss   : {avg_val:.4f}")
        print(f"  LR (head)  : {current_lr:.2e}")

        # ── AUC every N epochs ────────────────────────────
        auc_computed = (epoch % AUC_EVERY_N == 0 or epoch > NUM_EPOCHS - 10)
        if auc_computed:
            mean_auc, per_disease_aucs = compute_val_auc(model, val_loader, device)
        else:
            mean_auc, per_disease_aucs = best_auc, {}

        delta_str = ""
        if prev_auc is not None:
            delta = mean_auc - prev_auc
            arrow = "↑" if delta > 0 else "↓"
            delta_str = f" {arrow}{abs(delta):.4f}"
        print(f"  Mean AUC   : {mean_auc:.4f}{delta_str}  (ref: NIH baseline 0.8031 test)")
        prev_auc = mean_auc

        if epoch % 5 == 0 and per_disease_aucs:
            print(f"\n  Per-disease AUC — Epoch {epoch}:")
            for disease in TARGET_LABELS:
                if disease not in per_disease_aucs:
                    print(f"    -- {disease:<22} (skipped — soft-label only)")
                    continue
                auc = per_disease_aucs[disease]
                g = "✅" if auc >= 0.80 else "⚠️ " if auc >= 0.70 else "❌"
                print(f"    {g} {disease:<22} {auc:.4f}")

        # ── Checkpoint every N epochs ─────────────────────
        if epoch % CKPT_EVERY_N == 0:
            ck = f"chexpert_efficientnet_checkpoint_epoch{epoch}.pth"
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_auc"       : best_auc,
            }, ck)
            print(f"  Checkpoint : {ck}")
            old_ck = f"chexpert_efficientnet_checkpoint_epoch{epoch - 2 * CKPT_EVERY_N}.pth"
            if os.path.exists(old_ck):
                os.remove(old_ck)

        history.append({
            "epoch"     : epoch,
            "train_loss": avg_train,
            "val_loss"  : avg_val,
            "mean_auc"  : mean_auc,
            "lr"        : current_lr,
            **{f"auc_{d}": per_disease_aucs.get(d, None) for d in TARGET_LABELS}
        })
        pd.DataFrame(history).to_csv("chexpert_efficientnet_training_history.csv", index=False)

        # ── Best model + early stopping ───────────────────
        # Patience only counts on epochs where AUC was actually computed.
        # Without this, non-AUC epochs (mean_auc == best_auc, not >) would
        # increment patience every epoch, making effective patience ~3 checks.
        if mean_auc > best_auc:
            best_auc       = mean_auc
            patience_count = 0
            if best_model_name and os.path.exists(best_model_name):
                os.remove(best_model_name)
            best_model_name = f"chexpert_efficientnet_best_auc{mean_auc:.4f}_ep{epoch}.pth"
            torch.save(model.state_dict(), best_model_name)
            print(f"  ✅ Best model → {best_model_name}")
        elif auc_computed:
            patience_count += 1
            print(f"  ⚠️  No improvement ({patience_count}/{PATIENCE}) | Best: {best_auc:.4f}")
            if patience_count >= PATIENCE and epoch >= MIN_EPOCHS_STOP:
                print(f"\n🛑 Early stopping at epoch {epoch} | Best AUC: {best_auc:.4f}")
                break

    print(f"\n✅ Training complete!")
    print(f"   Best AUC        : {best_auc:.4f}")
    print(f"   Best model file : {best_model_name}")
    return best_auc, best_model_name


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print("=" * 65)
    print("   AyShCXR — CHEXPERT PLUS TRAINING RUN")
    print("   by Subhrakant Sethi & Ayush Singh")
    print("   Dataset  : CheXpert Plus (223,462 images)")
    print("   Backbone : EfficientNet-B4 at 380px")
    print("   GPU run  : Professor's A100 80GB")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  CHEXPERT_ROOT : {CHEXPERT_ROOT}")
    print(f"  CHEXPERT_CSV  : {CHEXPERT_CSV}")
    if not os.path.isdir(CHEXPERT_ROOT):
        print(f"❌ CHEXPERT_ROOT not found: {CHEXPERT_ROOT}")
        print(f"   Edit CHEXPERT_ROOT at the top of this file")
        return

    device = get_device()

    # ── VRAM check — auto-adjust batch size ──────────────
    global BATCH_SIZE, ACCUMULATION_STEPS
    if torch.cuda.is_available():
        free_vram, total_vram = [x / 1024**3 for x in torch.cuda.mem_get_info(0)]
        print(f"\n   VRAM: {free_vram:.1f}GB free / {total_vram:.1f}GB total")
        if total_vram >= 60:     # A100 80GB or H100
            BATCH_SIZE         = 128
            ACCUMULATION_STEPS = 1
        elif total_vram >= 24:   # RTX 3090 / A6000
            BATCH_SIZE         = 64
            ACCUMULATION_STEPS = 2
        elif total_vram >= 16:   # RTX 3080 / A5000
            BATCH_SIZE         = 32
            ACCUMULATION_STEPS = 4
        elif total_vram >= 10:   # RTX 3080 Ti
            BATCH_SIZE         = 16
            ACCUMULATION_STEPS = 8
        else:
            BATCH_SIZE         = 8
            ACCUMULATION_STEPS = 16
        print(f"✅ BATCH_SIZE={BATCH_SIZE}, ACCUMULATION_STEPS={ACCUMULATION_STEPS}")
        print(f"   Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")

    train_df, val_df, img_root = load_chexpert_data()
    if train_df is None:
        print("❌ Could not load CheXpert Plus data. Exiting.")
        return

    val_loader = DataLoader(
        CheXpertDataset(val_df, get_val_transform(), img_root),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    # ── Build model ───────────────────────────────────────
    model = create_efficientnet(num_classes=14).to(device)
    try:
        model = torch.compile(model)
        print("✅ torch.compile enabled (A100 — full kernel fusion)")
    except Exception as e:
        print(f"⚠️  torch.compile unavailable: {e}")

    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    # ── Auto-resume ───────────────────────────────────────
    start_epoch      = 1
    initial_best_auc = 0.0

    for ep in range(NUM_EPOCHS, 0, -1):
        ck = f"chexpert_efficientnet_checkpoint_epoch{ep}.pth"
        if os.path.exists(ck):
            print(f"\n⚠️  Found checkpoint: {ck}")
            if AUTO_RESUME:
                print("AUTO_RESUME=True — resuming automatically...")
                ans = "y"
            else:
                ans = input("Resume? (y/n): ").strip().lower()
            if ans == "y":
                checkpoint = torch.load(ck, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                    initial_best_auc = checkpoint["best_auc"]
                    start_epoch      = checkpoint["epoch"] + 1
                else:
                    model.load_state_dict(checkpoint)
                    start_epoch = ep + 1
                print(f"✅ Resuming from epoch {start_epoch} | Best AUC: {initial_best_auc:.4f}")
            else:
                print("✅ Starting fresh")
            break

    # ── Sanity check ──────────────────────────────────────
    print("\nRunning sanity check...")
    sample_ds = CheXpertDataset(train_df.head(4), get_val_transform(), img_root)
    sample_loader = DataLoader(sample_ds, batch_size=4, num_workers=0)
    imgs, labels = next(iter(sample_loader))
    with torch.no_grad():
        out = model(imgs.to(device))

    assert out.shape[1] == 14, f"❌ Wrong output shape: {out.shape}"
    probs = torch.sigmoid(out)
    assert probs.min() >= 0 and probs.max() <= 1, "❌ Sigmoid outputs out of range"

    print(f"  Input shape  : {imgs.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Output range : {probs.min():.3f} – {probs.max():.3f}")
    print(f"  Labels sample: {labels[0].numpy().round(2)}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  VRAM used    : {allocated:.2f}GB (batch of 4)")
    print(f"✅ Sanity check passed — EfficientNet-B4 at {IMG_SIZE}px working on CheXpert data")

    best_auc, best_model_name = train_loop(
        model, train_df, val_loader, device, optimizer, scheduler,
        start_epoch=start_epoch, initial_best_auc=initial_best_auc
    )

    print("\n" + "=" * 65)
    print("   CHEXPERT PLUS RUN COMPLETE")
    print(f"   Best AUC              : {best_auc:.4f}")
    print(f"   Best model file       : {best_model_name}")
    print(f"   History CSV           : chexpert_efficientnet_training_history.csv")
    print(f"   Indication texts      : chexpert_indication_texts.csv")
    print(f"   Radiologist benchmark : 0.778")
    print(f"   NIH DenseNet baseline : 0.8031 (test)")
    print()
    print("   NEXT STEP:")
    print("   1. Copy best checkpoint to safe name:")
    print(f"      cp {best_model_name} chexpert_efficientnet_BEST_safe.pth")
    print("   2. Run eval_chexpert.py to get CheXpert 5-disease benchmark AUC")
    print("   3. Run eval_and_save_preds.py --model <checkpoint> on NIH test set")
    print("      → Compare per-disease AUC vs NIH baseline 0.8031")
    print("   4. Use chexpert_indication_texts.csv for symptom fusion training")
    print("=" * 65)


if __name__ == "__main__":
    main()
