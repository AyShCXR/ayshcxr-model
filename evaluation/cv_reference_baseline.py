# cv_reference_baseline.py  —  RUN ON THE GPU SERVER (the last GPU night)
# Adds the REFERENCE model (CheXpert paper, Irvin & Rajpurkar 2019 = vanilla DenseNet-121)
# to your existing 3-fold CV, on the IDENTICAL folds/sample/recipe as cv_chexpert.py.
# Only the architecture differs (GAP-only + single Linear head, no GMP+GAP, no complex head)
# -> the measured gap IS your contribution. Appends one 'densenet121_vanilla' row to
# cv_results.csv. Everything else (seed, 90k subset, GroupKFold) is byte-for-byte the same,
# so your existing 3 rows stay valid on the same folds.
#   python cv_reference_baseline.py
#
# NOTE: keeps cv_chexpert.py untouched. Never reads/writes any .pth.

import os, numpy as np, pandas as pd
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"   # FIX: serial compile -> no inductor worker-pool deadlock under nohup
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')   # shm-safe (matches cv_chexpert)
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# ── IDENTICAL config to cv_chexpert.py (do not change — guarantees same folds) ───────────
CLEAN_CSV = "/workspace/chexpert_clean.csv"
RESULTS   = "/workspace/cv_results.csv"
N_FOLDS, EPOCHS, SUBSET, SEED = 3, 10, 90000, 42
ARCH_NAME = "densenet121_vanilla"            # the row label for the reference baseline
IMG_SIZE, BATCH = 380, 64
TARGET_LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
DISEASE_WEIGHTS = {"Enlarged Cardiomediastinum":3.5,"Cardiomegaly":1.4,"Lung Opacity":0.65,
    "Lung Lesion":2.8,"Edema":1.1,"Consolidation":2.6,"Pneumonia":4.5,"Atelectasis":1.35,
    "Pneumothorax":2.0,"Pleural Effusion":0.7,"Pleural Other":4.6,"Fracture":3.0,
    "Support Devices":0.55,"No Finding":3.0}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device} | REFERENCE = vanilla DenseNet-121 (CheXpert paper) | "
      f"img {IMG_SIZE} | batch {BATCH}")

# ── the REFERENCE architecture: standard torchvision DenseNet-121 head (GAP + Linear) ────
# This is exactly the CheXpert-paper style model: GlobalAvgPool -> one Linear(1024, 14).
# No GMP, no BN/Dropout/GELU complex head -> it lacks YOUR improvements on purpose.
class VanillaDenseNet(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        bb = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = bb.features
        in_f = bb.classifier.in_features          # 1024
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_f, n)      # plain single linear (head LR group)
    def forward(self, x):
        f = F.relu(self.features(x), inplace=True)
        return self.classifier(self.gap(f).flatten(1))

# ── IDENTICAL loss / transforms / dataset / train loop to cv_chexpert.py ─────────────────
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

def train_fold(tr_df, va_df):
    model = VanillaDenseNet().to(device)
    try: model = torch.compile(model)              # compile KEPT for speed; deadlock fixed by COMPILE_THREADS=1 (top)
    except Exception as e: print("compile skipped:", e, flush=True)
    tl = DataLoader(DS(tr_df, IMG_SIZE, True),  batch_size=BATCH,   shuffle=True,  num_workers=4,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(DS(va_df, IMG_SIZE, False), batch_size=BATCH*2, shuffle=False,
                    num_workers=0, pin_memory=True)
    bb_p, hd_p = [], []                                            # same backbone/head LR split
    for n, p in model.named_parameters():
        (hd_p if ("head" in n or "classifier" in n) else bb_p).append(p)
    opt = torch.optim.AdamW([{"params": bb_p, "lr": 6e-5, "weight_decay": 1e-5},
                             {"params": hd_p, "lr": 2e-4, "weight_decay": 1e-4}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    crit = FocalLoss(torch.tensor([DISEASE_WEIGHTS[d] for d in TARGET_LABELS], dtype=torch.float32))
    print("    training... first batch JIT-compiles (~2-3 min, GPU idle then is NORMAL)", flush=True)
    for ep in range(1, EPOCHS+1):
        model.train(); tot = 0.0; nb = 0
        for imgs, labs in tqdm(tl, desc=f"{ARCH_NAME} ep{ep:02d}", leave=False):
            imgs = imgs.to(device, non_blocking=True); labs = labs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'): loss = crit(model(imgs).float(), labs)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"    ep{ep:02d}/{EPOCHS} done | loss {tot/nb:.4f}", flush=True)   # life-sign each epoch
    return auc_of(model, vl)

# ── same df sampling + same GroupKFold (deterministic) => same folds as your existing rows ─
df = pd.read_csv(CLEAN_CSV)
if len(df) > SUBSET: df = df.sample(SUBSET, random_state=SEED).reset_index(drop=True)
print(f"CV on {len(df):,} images | {N_FOLDS}-fold | {EPOCHS} epochs/fold | reference baseline")

fold_aucs = []
for k, (tri, vai) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["patient_id"])):
    tr_df, va_df = df.iloc[tri], df.iloc[vai]
    ov = len(set(tr_df["patient_id"]) & set(va_df["patient_id"]))
    print(f"  fold {k+1}: train {len(tr_df):,} | val {len(va_df):,} | patient overlap {ov}")
    a = train_fold(tr_df, va_df)
    fold_aucs.append(a); print(f"  {ARCH_NAME} fold {k+1}/{N_FOLDS}: AUC {a:.4f}")
m, s = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
print(f"\n==> {ARCH_NAME} {N_FOLDS}-fold CV AUC: {m:.4f} +/- {s:.4f}")

# ── append (idempotent) to cv_results.csv, preserving your existing rows ──────────────────
cols = ["arch","cv_mean_auc","cv_std"] + [f"fold{i+1}" for i in range(N_FOLDS)]
row  = {"arch": ARCH_NAME, "cv_mean_auc": m, "cv_std": s,
        **{f"fold{i+1}": fold_aucs[i] for i in range(N_FOLDS)}}
if os.path.exists(RESULTS):
    out = pd.read_csv(RESULTS)
    out = out[out["arch"] != ARCH_NAME]                          # drop old vanilla row if re-running
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
else:
    print(f"  (note) {RESULTS} not found — writing a new file with only the baseline row;"
          f" re-add your existing rows before building the table.")
    out = pd.DataFrame([row], columns=cols)
out.to_csv(RESULTS, index=False)
print(f"saved {RESULTS}  — download this to your laptop, then run build_baseline_table.py")
