# chexpert_efficientnet_train.py
# AyShCXR — CheXpert Plus training. Proven recipe: 3-channel, GMP+GAP (CNN),
# Focal + LLRD, fast 18-epoch high-LR, patient-level split. shm-safe loaders.
#   python chexpert_efficientnet_train.py

import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')   # dgxhnode5 small /dev/shm
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

ARCH       = "efficientnet_b4"        # <-- only line that differs across the 3 files
CLEAN_CSV  = "/workspace/chexpert_clean.csv"
NUM_EPOCHS = 18
SEED, VAL_FRAC = 42, 0.08
TARGET_LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
DISEASE_WEIGHTS = {"Enlarged Cardiomediastinum":3.5,"Cardiomegaly":1.4,"Lung Opacity":0.65,
    "Lung Lesion":2.8,"Edema":1.1,"Consolidation":2.6,"Pneumonia":4.5,"Atelectasis":1.35,
    "Pneumothorax":2.0,"Pleural Effusion":0.7,"Pleural Other":4.6,"Fracture":3.0,
    "Support Devices":0.55,"No Finding":3.0}
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224 if ARCH == "rad_dino" else 380
BATCH    = 32 if ARCH == "rad_dino" else 64
print(f"Device {device} | ARCH {ARCH} | img {IMG_SIZE} | batch {BATCH}")

def complex_head(in_f, n=14):
    return nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4),
        nn.Linear(in_f, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n))

class CNNDualPool(nn.Module):                      # EfficientNet / DenseNet + GMP+GAP
    def __init__(self, arch, n=14):
        super().__init__()
        if arch == "efficientnet_b4":
            bb = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            self.features = bb.features; in_f = bb.classifier[1].in_features; self.post = nn.Identity()
        else:
            bb = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.features = bb.features; in_f = bb.classifier.in_features; self.post = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1); self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = complex_head(in_f, n)
    def forward(self, x):
        f = self.post(self.features(x))
        return self.classifier(self.gap(f).flatten(1) + self.gmp(f).flatten(1))

class RadDino(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
        self.head = complex_head(self.backbone.config.hidden_size, n)
    def forward(self, x):
        out = self.backbone(pixel_values=x)
        feat = out.pooler_output
        if feat is None: feat = out.last_hidden_state[:, 0]
        return self.head(feat)

def build_model():
    return RadDino() if ARCH == "rad_dino" else CNNDualPool(ARCH)

# ── data + patient split ────────────────────────────────────────────────────
df = pd.read_csv(CLEAN_CSV)
tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRAC, random_state=SEED).split(df, groups=df["patient_id"]))
train_df, val_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
ov = len(set(train_df["patient_id"]) & set(val_df["patient_id"]))
print(f"train {len(train_df):,} | val {len(val_df):,} | patient overlap {ov}")

norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
train_tf = T.Compose([T.Resize((IMG_SIZE+32, IMG_SIZE+32)), T.RandomCrop(IMG_SIZE),
    T.RandomRotation(15), T.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
    T.RandomPerspective(0.1, p=0.3), T.ColorJitter(0.2,0.2),
    T.GaussianBlur(3, sigma=(0.1,1.0)), T.ToTensor(), norm])
val_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])

class DS(Dataset):
    def __init__(self, d, tf): self.d=d.reset_index(drop=True); self.tf=tf
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128,128,128))
        return self.tf(img), torch.tensor(r[TARGET_LABELS].to_numpy(dtype="float32"))

train_loader = DataLoader(DS(train_df, train_tf), batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)   # non-persistent
val_loader   = DataLoader(DS(val_df, val_tf), batch_size=BATCH*2, shuffle=False,
                          num_workers=0, pin_memory=True)                   # shm-safe

model = build_model().to(device)
try:    model = torch.compile(model); print("torch.compile applied")
except Exception as e: print("compile skipped:", e)

head_keys = ("head", "classifier")
bb_p, hd_p = [], []
for n, p in model.named_parameters():
    (hd_p if any(k in n for k in head_keys) else bb_p).append(p)
bb_lr = 3e-5 if ARCH == "rad_dino" else 6e-5
opt = torch.optim.AdamW([{"params": bb_p, "lr": bb_lr, "weight_decay": 1e-5},
                         {"params": hd_p, "lr": 2e-4, "weight_decay": 1e-4}])
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-7)
scaler = torch.amp.GradScaler('cuda')

class FocalLoss(nn.Module):
    def __init__(self, g=2.0, a=0.75, s=0.1, w=None): super().__init__(); self.g=g; self.a=a; self.s=s; self.w=w
    def forward(self, x, t):
        x=x.float(); t=t.float().clamp(0,1); ts=t*(1-self.s)+0.5*self.s
        bce=F.binary_cross_entropy_with_logits(x, ts, reduction='none').clamp(max=50)
        pt=torch.exp(-bce); fl=self.a*(1-pt)**self.g*bce
        if self.w is not None: fl=fl*self.w.to(x.device)
        return fl.mean()
crit = FocalLoss(w=torch.tensor([DISEASE_WEIGHTS[d] for d in TARGET_LABELS], dtype=torch.float32))

def run_val():
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for imgs, labs in val_loader:
            with torch.amp.autocast('cuda'): p = torch.sigmoid(model(imgs.to(device, non_blocking=True)))
            P.append(p.float().cpu().numpy()); Y.append(labs.numpy())
    P, Y = np.vstack(P), np.vstack(Y); per = {}
    for i, d in enumerate(TARGET_LABELS):
        if len(np.unique(Y[:, i])) > 1: per[d] = roc_auc_score(Y[:, i], P[:, i])
    return float(np.mean(list(per.values()))), per

print("="*55 + f"\n  CheXpert {ARCH} — fast proven recipe\n" + "="*55)
best = 0.0
for ep in range(1, NUM_EPOCHS+1):
    model.train(); tot = 0.0; nb = 0
    for imgs, labs in tqdm(train_loader, desc=f"{ARCH} ep{ep:02d}"):
        imgs = imgs.to(device, non_blocking=True); labs = labs.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'): logits = model(imgs)
        loss = crit(logits.float(), labs)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); tot += loss.item(); nb += 1
    sched.step()
    auc, per = run_val()
    flag = "  BEST" if auc > best else ""
    print(f"  ep{ep:02d} | loss {tot/nb:.4f} | val-AUC {auc:.4f}{flag}")
    if auc > best:
        best = auc
        torch.save(model.state_dict(), f"/workspace/chexpert_{ARCH}_best_auc{auc:.4f}_ep{ep}.pth")
        pd.DataFrame({"disease": list(per.keys()), "auc": list(per.values())}) \
          .to_csv(f"/workspace/chexpert_{ARCH}_per_disease.csv", index=False)
print(f"\nDONE {ARCH} | best val-AUC {best:.4f}")
