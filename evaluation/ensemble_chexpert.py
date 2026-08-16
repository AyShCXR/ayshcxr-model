# ensemble_chexpert.py
# Ensembles the 3 new CheXpert models (chexpert_<arch>_best_auc*.pth) on the same
# val split and reports per-model + ensemble AUC. Matches the GMP+GAP architecture.
#   python ensemble_chexpert.py

import glob, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

CLEAN_CSV = "/workspace/chexpert_clean.csv"
SEED, VAL_FRAC = 42, 0.08
TARGET_LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def complex_head(in_f, n=14):
    return nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4),
        nn.Linear(in_f, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n))

class CNNDualPool(nn.Module):
    def __init__(self, arch, n=14):
        super().__init__()
        if arch == "efficientnet_b4":
            bb = models.efficientnet_b4(weights=None)
            self.features = bb.features; in_f = bb.classifier[1].in_features; self.post = nn.Identity()
        else:
            bb = models.densenet121(weights=None)
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
        out = self.backbone(pixel_values=x); feat = out.pooler_output
        if feat is None: feat = out.last_hidden_state[:, 0]
        return self.head(feat)

CKPTS = {  # arch: (filename glob, input size)
    "efficientnet_b4": ("/workspace/chexpert_efficientnet_b4_best_auc*.pth", 380),
    "densenet121":     ("/workspace/chexpert_densenet121_best_auc*.pth", 380),
    "rad_dino":        ("/workspace/chexpert_rad_dino_best_auc*.pth", 224),
}
def best_ckpt(pattern):
    files = glob.glob(pattern)
    return max(files, key=lambda f: float(f.split("auc")[1].split("_")[0])) if files else None

# same val split as training
df = pd.read_csv(CLEAN_CSV)
_, va = next(GroupShuffleSplit(1, test_size=VAL_FRAC, random_state=SEED).split(df, groups=df["patient_id"]))
val_df = df.iloc[va].reset_index(drop=True)
Y = val_df[TARGET_LABELS].to_numpy(dtype="float32")
norm = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])

class ValDS(Dataset):
    def __init__(self, size): self.tf = T.Compose([T.Resize((size,size)), T.ToTensor(), norm]); self.size=size
    def __len__(self): return len(val_df)
    def __getitem__(self, i):
        r = val_df.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (self.size,self.size), (128,128,128))
        return self.tf(img)

def mean_auc(P):
    return float(np.mean([roc_auc_score(Y[:,i], P[:,i]) for i in range(14) if len(np.unique(Y[:,i]))>1]))

def predict(arch, ckpt, size):
    model = (RadDino() if arch == "rad_dino" else CNNDualPool(arch)).to(device).eval()
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt.split('/')[-1]} | missing={len(miss)} unexpected={len(unexp)}")
    loader = DataLoader(ValDS(size), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    P = []
    with torch.no_grad():
        for imgs in loader:
            with torch.amp.autocast('cuda'): p = torch.sigmoid(model(imgs.to(device, non_blocking=True)))
            P.append(p.float().cpu().numpy())
    return np.vstack(P)

probs = {}
print("="*55)
for arch, (pat, size) in CKPTS.items():
    ck = best_ckpt(pat)
    if ck is None: print(f"{arch}: NO checkpoint"); continue
    P = predict(arch, ck, size); probs[arch] = P
    print(f"{arch:<18} mean AUC {mean_auc(P):.4f}")

if len(probs) < 2: print("\nNeed >=2 models."); raise SystemExit
ens = np.mean(list(probs.values()), axis=0)
print("="*55 + f"\nENSEMBLE of {len(probs)} models: mean AUC {mean_auc(ens):.4f}\n" + "="*55)
rows = []
for i, d in enumerate(TARGET_LABELS):
    if len(np.unique(Y[:,i])) > 1:
        a = roc_auc_score(Y[:,i], ens[:,i]); rows.append((d, a)); print(f"  {d:<28} {a:.4f}")
pd.DataFrame(rows, columns=["disease","auc"]).to_csv("/workspace/chexpert_ensemble_per_disease.csv", index=False)
print("\nsaved chexpert_ensemble_per_disease.csv")
