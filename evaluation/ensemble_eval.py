# ensemble_eval.py
# AyShCXR — averages the per-model predictions on the SAME val split and reports
# the ensemble AUC (usually higher than any single model). Defensible, no leakage.
#
#   python ensemble_eval.py
#
# Uses each model's best_<arch>.pth (and the EfficientNet checkpoint) at its own
# input resolution; val rows are identical & in identical order across models
# (GroupShuffleSplit seed 42, shuffle=False), so probabilities align row-for-row.

import os, numpy as np, pandas as pd, torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from train_model import build_model, DS, load_ckpt, TARGET_LABELS, CLEAN_CSV, VAL_FRAC, SEED, device

# arch -> checkpoint file (first that exists is used)
CKPTS = {
    "efficientnet_b4": ["best_efficientnet_b4.pth", "chexpert_best_auc0.8164_ep11.pth"],
    "densenet121":     ["best_densenet121.pth"],
    "rad_dino":        ["best_rad_dino.pth"],
    "swin_t":          ["best_swin_t.pth"],
}
OUT_DIR = "/workspace"
norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

# same val split as training
df = pd.read_csv(CLEAN_CSV)
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
_, va_idx = next(gss.split(df, groups=df["patient_id"]))
val_df = df.iloc[va_idx].reset_index(drop=True)
Y = val_df[TARGET_LABELS].to_numpy(dtype="float32")

def mean_auc(P):
    aucs = [roc_auc_score(Y[:,i], P[:,i]) for i in range(14) if len(np.unique(Y[:,i]))>1]
    return float(np.mean(aucs))

def predict(arch, ckpt):
    model, size = build_model(arch)
    model = model.to(device).eval()
    load_ckpt(model, ckpt)
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])
    loader = DataLoader(DS(val_df, tf, size), batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)
    P = []
    with torch.no_grad():
        for imgs, _ in loader:
            with torch.amp.autocast('cuda'):
                p = torch.sigmoid(model(imgs.to(device, non_blocking=True)))
            P.append(p.float().cpu().numpy())
    return np.vstack(P)

probs = {}
print("="*55)
for arch, candidates in CKPTS.items():
    ckpt = next((c for c in candidates if os.path.exists(os.path.join(OUT_DIR, c))), None)
    if ckpt is None:
        print(f"{arch:<18} — no checkpoint, skipping"); continue
    P = predict(arch, os.path.join(OUT_DIR, ckpt))
    probs[arch] = P
    print(f"{arch:<18} mean AUC {mean_auc(P):.4f}   ({ckpt})")

if len(probs) < 2:
    print("\nNeed >=2 models for an ensemble. Train more first."); raise SystemExit

ens = np.mean(list(probs.values()), axis=0)          # simple average of probabilities
print("="*55)
print(f"ENSEMBLE of {len(probs)} models: mean AUC {mean_auc(ens):.4f}")
print("="*55)
rows = []
for i, d in enumerate(TARGET_LABELS):
    if len(np.unique(Y[:,i])) > 1:
        a = roc_auc_score(Y[:,i], ens[:,i]); rows.append((d, a))
        print(f"   {d:<28} {a:.4f}")
pd.DataFrame(rows, columns=["disease","auc"]).to_csv(os.path.join(OUT_DIR,"per_disease_ensemble.csv"), index=False)
print("\nSaved per_disease_ensemble.csv")
