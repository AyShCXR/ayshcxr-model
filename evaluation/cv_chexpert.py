# cv_chexpert.py
# TRUE 3-fold cross-validation (fresh model retrained per fold) for the 3 CheXpert
# architectures. Fast config to fit the GPU window: 3 folds, 10 epochs, 90k subset,
# patient-grouped folds (no leakage). Reports mean +/- std AUC per model.
#   python cv_chexpert.py
#
# NOTE: reduced epochs/subset -> CV AUC reads a bit lower than the full-model test AUC.
# That is expected; report it honestly.

import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

CLEAN_CSV = "/workspace/chexpert_clean.csv"
N_FOLDS, EPOCHS, SUBSET, SEED = 3, 10, 90000, 42
TARGET_LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
DISEASE_WEIGHTS = {"Enlarged Cardiomediastinum":3.5,"Cardiomegaly":1.4,"Lung Opacity":0.65,
    "Lung Lesion":2.8,"Edema":1.1,"Consolidation":2.6,"Pneumonia":4.5,"Atelectasis":1.35,
    "Pneumothorax":2.0,"Pleural Effusion":0.7,"Pleural Other":4.6,"Fracture":3.0,
    "Support Devices":0.55,"No Finding":3.0}
ARCHS = {"efficientnet_b4": 380, "densenet121": 380, "rad_dino": 224}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def complex_head(in_f, n=14):
    return nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4),
        nn.Linear(in_f, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n))
class CNNDualPool(nn.Module):
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
        f = self.post(self.features(x)); return self.classifier(self.gap(f).flatten(1) + self.gmp(f).flatten(1))
class RadDino(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
        self.head = complex_head(self.backbone.config.hidden_size, n)
    def forward(self, x):
        out = self.backbone(pixel_values=x); feat = out.pooler_output
        if feat is None: feat = out.last_hidden_state[:, 0]
        return self.head(feat)
def build(arch): return RadDino() if arch == "rad_dino" else CNNDualPool(arch)

class FocalLoss(nn.Module):
    def __init__(self, w): super().__init__(); self.w = w
    def forward(self, x, t):
        x=x.float(); t=t.float().clamp(0,1); ts=t*0.9+0.05
        bce=F.binary_cross_entropy_with_logits(x, ts, reduction='none').clamp(max=50)
        pt=torch.exp(-bce); return (0.75*(1-pt)**2.0*bce*self.w.to(x.device)).mean()

norm = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
def tfs(size, train):
    if train:
        return T.Compose([T.Resize((size+32,size+32)), T.RandomCrop(size), T.RandomRotation(15),
            T.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)), T.ColorJitter(0.2,0.2),
            T.ToTensor(), norm])
    return T.Compose([T.Resize((size,size)), T.ToTensor(), norm])
class DS(Dataset):
    def __init__(self, d, size, train): self.d=d.reset_index(drop=True); self.tf=tfs(size,train); self.size=size
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (self.size,self.size), (128,128,128))
        return self.tf(img), torch.tensor(r[TARGET_LABELS].to_numpy(dtype="float32"))

def auc_of(model, loader):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            with torch.amp.autocast('cuda'): p = torch.sigmoid(model(imgs.to(device, non_blocking=True)))
            P.append(p.float().cpu().numpy()); Y.append(labs.numpy())
    P, Y = np.vstack(P), np.vstack(Y)
    return float(np.mean([roc_auc_score(Y[:,i], P[:,i]) for i in range(14) if len(np.unique(Y[:,i]))>1]))

def train_fold(arch, size, tr_df, va_df, batch):
    model = build(arch).to(device)
    try: model = torch.compile(model)
    except Exception: pass
    tl = DataLoader(DS(tr_df, size, True), batch_size=batch, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)              # non-persistent
    vl = DataLoader(DS(va_df, size, False), batch_size=batch*2, shuffle=False, num_workers=4, pin_memory=True)
    bb_lr = 3e-5 if arch == "rad_dino" else 6e-5
    bb_p, hd_p = [], []
    for n, p in model.named_parameters():
        (hd_p if ("head" in n or "classifier" in n) else bb_p).append(p)
    opt = torch.optim.AdamW([{"params": bb_p, "lr": bb_lr, "weight_decay": 1e-5},
                             {"params": hd_p, "lr": 2e-4, "weight_decay": 1e-4}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    crit = FocalLoss(torch.tensor([DISEASE_WEIGHTS[d] for d in TARGET_LABELS], dtype=torch.float32))
    for ep in range(1, EPOCHS+1):                 # train only — NO per-epoch validation
        model.train()
        for imgs, labs in tqdm(tl, desc=f"{arch} ep{ep:02d}", leave=False):
            imgs = imgs.to(device, non_blocking=True); labs = labs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'): loss = crit(model(imgs).float(), labs)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        sched.step()
    return auc_of(model, vl)                       # validate ONCE at the end (this was the slow part)

df = pd.read_csv(CLEAN_CSV)
if len(df) > SUBSET: df = df.sample(SUBSET, random_state=SEED).reset_index(drop=True)
print(f"CV on {len(df):,} images | {N_FOLDS}-fold | {EPOCHS} epochs/fold")

results = []
for arch, size in ARCHS.items():
    batch = 32 if arch == "rad_dino" else 64
    fold_aucs = []
    for k, (tri, vai) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["patient_id"])):
        a = train_fold(arch, size, df.iloc[tri], df.iloc[vai], batch)
        fold_aucs.append(a); print(f"  {arch} fold {k+1}/{N_FOLDS}: AUC {a:.4f}")
    m, s = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    print(f"==> {arch} {N_FOLDS}-fold CV AUC: {m:.4f} +/- {s:.4f}\n")
    results.append([arch, m, s] + fold_aucs)

pd.DataFrame(results, columns=["arch","cv_mean_auc","cv_std"] + [f"fold{i+1}" for i in range(N_FOLDS)]) \
  .to_csv("/workspace/cv_results.csv", index=False)
print("saved /workspace/cv_results.csv")
