# build_and_train_demo.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi & Ayush Singh
#
# EfficientNet-B4 at 380px — Professor GPU Run
#
# Changes from DenseNet run:
#   ✅ Architecture: DenseNet-121 → EfficientNet-B4
#   ✅ Resolution: 224px → 380px (native resolution for B4)
#   ✅ Batch size: 32 → 16 (larger images need more VRAM)
#   ✅ Checkpoint names updated to efficientnet_*
#
# Everything else kept identical:
#   ✅ Focal Loss (gamma=2.0, alpha=0.25) + Label Smoothing (0.1)
#   ✅ Multilabel stratified split
#   ✅ AUC-based early stopping, patience=15, epochs=80
#   ✅ Layer-wise LR: backbone 1e-5, head 3e-5
#   ✅ Strong augmentation pipeline
#   ✅ Gradient clipping
#   ✅ Per-disease loss weights
#   ✅ Per-epoch AUC logging
#   ✅ Best model filename includes AUC — prevents overwrite

import os
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

# ── Device ────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory
        print(f"✅ GPU: {name}")
        print(f"   VRAM: {vram/1024**3:.1f} GB")
        vram_gb = vram / 1024**3
        if vram_gb >= 40:
            print(f"   Suggested BATCH_SIZE for EfficientNet-B4 at 380px: 64")
        elif vram_gb >= 24:
            print(f"   Suggested BATCH_SIZE for EfficientNet-B4 at 380px: 32")
        elif vram_gb >= 16:
            print(f"   Suggested BATCH_SIZE for EfficientNet-B4 at 380px: 24")
        elif vram_gb >= 12:
            print(f"   Suggested BATCH_SIZE for EfficientNet-B4 at 380px: 16")
        else:
            print(f"   Suggested BATCH_SIZE for EfficientNet-B4 at 380px: 8")
            print(f"   ⚠️  Less than 12GB VRAM — may be tight at 380px")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU — using CPU")
    return device

# ── CONFIG ────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
BATCH_SIZE   = 32     # ← Restored; AMP reduces VRAM pressure
NUM_WORKERS  = 8      # ← Increased for heavier transforms
NUM_EPOCHS   = 60     # ← Reduced; early stopping fires before this anyway
IMG_SIZE     = 380    # ← EfficientNet-B4 native resolution
LR_HEAD      = 3e-5
LR_BACKBONE  = 1e-5
PATIENCE           = 8        # AUC checked every 3 epochs; effective ~24 epochs
MIN_EPOCHS         = 20       # No early stopping before this epoch
VAL_SPLIT          = 0.08     # 3,440 more training images vs 0.12
LABEL_SMOOTH       = 0.1
USE_AMP            = True     # Mixed precision — 2-3x speed boost
AUC_EVERY_N        = 3        # Compute AUC every 3 epochs
CKPT_EVERY_N       = 5        # Save checkpoint every 5 epochs
AUTO_RESUME        = True     # Auto-resume on non-interactive server
ACCUMULATION_STEPS = 4        # Effective batch = BATCH_SIZE * 4 = 128

NIH_LABELS_CSV  = "nih_full_labels.csv"
NIH_TRAIN_LIST  = "train_val_list.txt"

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
    "Pneumonia"         : 2.00,   # most dangerous for rural India use case
    "Pneumothorax"      : 0.87,
    "Consolidation"     : 0.93,
    "Edema"             : 1.33,
    "Emphysema"         : 1.27,
    "Fibrosis"          : 1.56,
    "Pleural Thickening": 1.10,   # recalculate after new CSV generated
    "Hernia"            : 3.00,
}

# ── Dataset ───────────────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row["full_path"]
        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")
            img = Image.new("L", (IMG_SIZE, IMG_SIZE), 128)
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[list(TARGET_LABELS)].to_numpy(dtype="float32"))
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
        pt       = torch.exp(-bce_loss)
        focal    = self.alpha * (1.0 - pt) ** self.gamma * bce_loss
        if self.disease_weights is not None:
            focal = focal * self.disease_weights.to(inputs.device)
        return focal.mean()

# ── EfficientNet-B4 ───────────────────────────────────────────────────────
def create_efficientnet(num_classes=14):
    print("Building EfficientNet-B4 at 380px...")
    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )

    # Convert first conv from 3-channel to 1-channel
    # Average RGB weights to preserve ImageNet transfer learning
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

    # Classification head
    # EfficientNet-B4 outputs 1792 features before classifier
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

# ── Optimizer ─────────────────────────────────────────────────────────────
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
        {"params": early_params,  "lr": 1e-6,       "weight_decay": 1e-5},
        {"params": middle_params, "lr": 5e-6,       "weight_decay": 1e-5},
        {"params": late_params,   "lr": LR_BACKBONE, "weight_decay": 1e-5},
        {"params": head_params,   "lr": LR_HEAD,     "weight_decay": 1e-4},
    ])

# ── Transforms ────────────────────────────────────────────────────────────
def get_train_transform():
    return T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomCrop(IMG_SIZE),
        # No horizontal flip — heart is left-sided; flip creates dextrocardia pattern
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # simulates cheap scanner blur
        T.RandAugment(num_ops=2, magnitude=9),             # replaces AutoContrast + Equalize
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229]),
        T.RandomErasing(p=0.1, value=0.485),               # fill with mean tissue value
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
        if len(np.unique(y_true)) < 2:
            continue
        try:
            auc_dict[disease] = roc_auc_score(y_true, all_preds[:, i])
        except Exception as e:
            print(f"⚠️  AUC failed for {disease}: {e}")
    mean_auc = float(np.mean(list(auc_dict.values()))) if auc_dict else 0.0
    return mean_auc, auc_dict

# ── Load NIH ──────────────────────────────────────────────────────────────
def load_nih_data():
    print("\n── Loading NIH ChestX-ray14 ──────────────────────")
    if not os.path.exists(NIH_LABELS_CSV):
        print(f"❌ {NIH_LABELS_CSV} not found!")
        return None
    df = pd.read_csv(NIH_LABELS_CSV)
    missing = [c for c in TARGET_LABELS if c not in df.columns]
    if missing:
        print(f"❌ Missing columns in CSV: {missing}")
        return None
    print(f"Total NIH images : {len(df):,}")
    if os.path.exists(NIH_TRAIN_LIST):
        with open(NIH_TRAIN_LIST) as f:
            train_images = set(line.strip() for line in f)
        df = df[df["Image Index"].isin(train_images)].reset_index(drop=True)
        print(f"NIH train pool   : {len(df):,}")
    df["label_count"] = df[list(TARGET_LABELS)].sum(axis=1)
    df["source"]      = "nih"
    return df

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
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    best_auc        = initial_best_auc
    patience_count  = 0
    best_model_name = None
    history         = []
    prev_auc        = None

    train_loader = DataLoader(
        ChestXrayDataset(train_df, get_train_transform()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )

    print("\n" + "=" * 65)
    print("   AyShCXR — EFFICIENTNET-B4 380px TRAINING RUN")
    print(f"   Training images  : {len(train_df):,}")
    print(f"   Val images       : {len(val_loader.dataset):,}")
    print(f"   Architecture     : EfficientNet-B4 (19M parameters)")
    print(f"   Loss             : Focal Loss (gamma=2.0, alpha=0.75)")
    print(f"   Image size       : {IMG_SIZE}px (B4 native resolution)")
    print(f"   Epochs           : {NUM_EPOCHS} max, patience={PATIENCE}")
    print(f"   Batch size       : {BATCH_SIZE}  (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"   AMP              : {'enabled' if USE_AMP else 'disabled'}")
    print(f"   Resuming from    : epoch {start_epoch}, best AUC {best_auc:.4f}")
    print(f"   DenseNet baseline: 0.8339")
    print("=" * 65)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        print(f"\nEpoch {epoch:02d}/{NUM_EPOCHS}  |  {IMG_SIZE}px  |  {len(train_df):,} images")

        # ── Train ─────────────────────────────────────────
        model.train()
        train_loss, train_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc="  Train")):
            imgs, labels = imgs.to(device), labels.to(device)

            # Mixup — simulates inter-patient variation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(imgs.size(0), device=device)
            imgs   = lam * imgs   + (1 - lam) * imgs[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs), labels) / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                loss = criterion(model(imgs), labels) / ACCUMULATION_STEPS
                loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ACCUMULATION_STEPS
            train_batches += 1

        avg_train = train_loss / train_batches
        scheduler.step()

        # ── Val loss (every other epoch — informational only) ─
        if epoch % 2 != 0:
            model.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    val_loss += criterion(model(imgs), labels).item()
                    val_batches += 1
            avg_val = val_loss / val_batches
        else:
            avg_val = 0.0

        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"  Train Loss : {avg_train:.4f}")
        if avg_val > 0:
            print(f"  Val Loss   : {avg_val:.4f}")
        print(f"  LR (head)  : {current_lr:.2e}")

        # ── AUC every N epochs (or last 10 epochs) ────────
        if epoch % AUC_EVERY_N == 0 or epoch > NUM_EPOCHS - 10:
            mean_auc, per_disease_aucs = compute_val_auc(model, val_loader, device)
        else:
            mean_auc, per_disease_aucs = best_auc, {}

        delta_str = ""
        if prev_auc is not None:
            delta = mean_auc - prev_auc
            arrow = "↑" if delta > 0 else "↓"
            delta_str = f" {arrow}{abs(delta):.4f}"
        print(f"  Mean AUC   : {mean_auc:.4f}{delta_str}  (DenseNet baseline: 0.8339)")
        prev_auc = mean_auc

        if epoch % 5 == 0 and per_disease_aucs:
            print(f"\n  Per-disease AUC — Epoch {epoch}:")
            for disease, auc in per_disease_aucs.items():
                g = "✅" if auc >= 0.80 else "⚠️ " if auc >= 0.70 else "❌"
                print(f"    {g} {disease:<22} {auc:.4f}")

        # ── Checkpoint every N epochs ─────────────────────
        if epoch % CKPT_EVERY_N == 0:
            ck = f"efficientnet_checkpoint_epoch{epoch}.pth"
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_auc"       : best_auc,
            }, ck)
            print(f"  Checkpoint : {ck}")
            old_ck = f"efficientnet_checkpoint_epoch{epoch - 2 * CKPT_EVERY_N}.pth"
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
        pd.DataFrame(history).to_csv("efficientnet_training_history.csv", index=False)

        # ── Early stopping ────────────────────────────────
        # Always reset patience when AUC improves.
        # Model saving + stopping guarded by MIN_EPOCHS.
        if mean_auc > best_auc:
            best_auc       = mean_auc
            patience_count = 0
            if epoch >= MIN_EPOCHS:
                if best_model_name and os.path.exists(best_model_name):
                    os.remove(best_model_name)
                best_model_name = f"efficientnet_best_auc{mean_auc:.4f}_ep{epoch}.pth"
                torch.save(model.state_dict(), best_model_name)
                print(f"  ✅ Best model → {best_model_name}")
            else:
                print(f"  ✅ AUC improved to {mean_auc:.4f} (model saving starts at epoch {MIN_EPOCHS})")
        else:
            patience_count += 1
            print(f"  ⚠️  No improvement ({patience_count}/{PATIENCE}) | Best: {best_auc:.4f}")
            if patience_count >= PATIENCE and epoch >= MIN_EPOCHS:
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
    print("   AyShCXR — EFFICIENTNET-B4 380px TRAINING RUN")
    print("   by Subhrakant Sethi & Ayush Singh")
    print("   Target: Beat DenseNet baseline of 0.8339")
    print("=" * 65)

    device = get_device()

    # ── VRAM check — auto-adjust batch size ──────────────
    global BATCH_SIZE
    if torch.cuda.is_available():
        free_vram, total_vram = [x / 1024**3 for x in torch.cuda.mem_get_info(0)]
        print(f"   VRAM: {free_vram:.1f}GB free / {total_vram:.1f}GB total")
        if free_vram < 10:
            BATCH_SIZE = 8
            print(f"⚠️  Low free VRAM — auto-reducing BATCH_SIZE to 8")
        elif free_vram < 16:
            BATCH_SIZE = 16
        elif free_vram < 24:
            BATCH_SIZE = 24
        else:
            BATCH_SIZE = 32
        print(f"✅ BATCH_SIZE set to {BATCH_SIZE}")

    nih_df = load_nih_data()
    if nih_df is None:
        print("❌ Could not load NIH data.")
        return

    print("\nBuilding stratified val split...")
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        for train_idx, val_idx in msss.split(nih_df, nih_df[list(TARGET_LABELS)].values):
            train_df = nih_df.iloc[train_idx].reset_index(drop=True)
            val_df   = nih_df.iloc[val_idx].reset_index(drop=True)
        print("✅ Multilabel stratified split applied")
    except ImportError:
        print("⚠️  WARNING: iterative-stratification not installed!")
        print("   Install with: pip install iterative-stratification")
        print("   Using random split — rare disease AUC may be unreliable")
        ans = input("   Continue with random split? (y/n): ").strip().lower()
        if ans != "y":
            return
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            nih_df, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

    print(f"\nDataset summary:")
    print(f"  Train : {len(train_df):,} images")
    print(f"  Val   : {len(val_df):,} images")

    val_loader = DataLoader(
        ChestXrayDataset(val_df, get_val_transform()),
        batch_size=BATCH_SIZE * 2,   # no gradients during val — can double batch
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )

    # ── Build model ───────────────────────────────────────
    model = create_efficientnet(num_classes=14).to(device)
    try:
        model = torch.compile(model)
        print("✅ torch.compile applied (PyTorch 2.0+)")
    except Exception:
        print("⚠️  torch.compile not available — skipping")

    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-7
    )

    # ── Resume check ──────────────────────────────────────
    start_epoch      = 1
    initial_best_auc = 0.0

    for ep in range(NUM_EPOCHS, 0, -1):
        ck = f"efficientnet_checkpoint_epoch{ep}.pth"
        if os.path.exists(ck):
            print(f"\n⚠️  Found checkpoint: {ck}")
            if AUTO_RESUME:
                ans = "y"
                print("AUTO_RESUME=True — resuming automatically...")
            else:
                ans = input("Resume from this checkpoint? (y/n): ").strip().lower()
            if ans == "y":
                checkpoint = torch.load(ck, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                    initial_best_auc = checkpoint["best_auc"]
                    start_epoch      = checkpoint["epoch"] + 1
                else:
                    # Legacy checkpoint — model state dict only
                    model.load_state_dict(checkpoint)
                    start_epoch = ep + 1
                print(f"✅ Resuming from epoch {start_epoch} | Best AUC: {initial_best_auc:.4f}")
            else:
                print("✅ Starting fresh")
            break

    # ── Sanity check ──────────────────────────────────────
    print("\nRunning sanity check...")
    sample_ds = ChestXrayDataset(train_df.head(BATCH_SIZE), get_val_transform())
    imgs, _   = next(iter(DataLoader(sample_ds, batch_size=BATCH_SIZE, num_workers=0)))
    with torch.no_grad():
        out = model(imgs.to(device))

    assert imgs.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE), \
        f"❌ Wrong input shape: {imgs.shape}"
    assert out.shape[1] == 14, \
        f"❌ Wrong output shape: {out.shape}"
    probs = torch.sigmoid(out)
    assert probs.min() >= 0 and probs.max() <= 1, \
        "❌ Sigmoid outputs out of range"

    print(f"  Input shape  : {imgs.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Output range : {probs.min():.3f} – {probs.max():.3f}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  VRAM used    : {allocated:.2f}GB (batch of {BATCH_SIZE})")
    print(f"✅ Sanity check passed — EfficientNet-B4 at {IMG_SIZE}px working")

    best_auc, best_model_name = train_loop(
        model, train_df, val_loader, device, optimizer, scheduler,
        start_epoch=start_epoch, initial_best_auc=initial_best_auc
    )

    print("\n" + "=" * 65)
    print("   TRAINING COMPLETE")
    print(f"   Best AUC              : {best_auc:.4f}")
    print(f"   Best model file       : {best_model_name}")
    print(f"   Radiologist benchmark : 0.778")
    print(f"   DenseNet baseline     : 0.8339")
    if best_auc > 0.8339:
        print(f"   ✅ Beat DenseNet by +{best_auc - 0.8339:.4f}")
    elif best_auc > 0.778:
        print(f"   ✅ Above radiologist benchmark by +{best_auc - 0.778:.4f}")
    print("=" * 65)
    print("\nNext step: python eval_and_save_preds.py")

if __name__ == "__main__":
    main()