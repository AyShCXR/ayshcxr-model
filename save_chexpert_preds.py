# save_chexpert_preds.py
# Runs the 3 CheXpert models on TRAIN / VAL / TEST splits and saves predictions +
# labels to chexpert_preds.npz (a few MB). Download that file, then build the whole
# metrics table + ROC plots + 5-fold CV on your laptop — no GPU/images needed.
#   python save_chexpert_preds.py

import glob, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import GroupShuffleSplit

CLEAN_CSV = "/workspace/chexpert_clean.csv"
SEED = 42
TRAIN_SAMPLE = 20000          # sample train for speed (representative for the table)
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
            bb = models.efficientnet_b4(weights=None); self.features = bb.features
            in_f = bb.classifier[1].in_features; self.post = nn.Identity()
        else:
            bb = models.densenet121(weights=None); self.features = bb.features
            in_f = bb.classifier.in_features; self.post = nn.ReLU(inplace=True)
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

CKPTS = {"efficientnet_b4": ("/workspace/chexpert_efficientnet_b4_best_auc*.pth", 380),
         "densenet121":     ("/workspace/chexpert_densenet121_best_auc*.pth", 380),
         "rad_dino":        ("/workspace/chexpert_rad_dino_best_auc*.pth", 224)}
def best_ckpt(p):
    f = glob.glob(p); return max(f, key=lambda x: float(x.split("auc")[1].split("_")[0])) if f else None

# splits: train(92%) + holdout(8%); holdout -> val(4%) + test(4%), all patient-level
df = pd.read_csv(CLEAN_CSV)
tr, hold = next(GroupShuffleSplit(1, test_size=0.08, random_state=SEED).split(df, groups=df["patient_id"]))
train_df, hold_df = df.iloc[tr].reset_index(drop=True), df.iloc[hold].reset_index(drop=True)
v, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(hold_df, groups=hold_df["patient_id"]))
val_df, test_df = hold_df.iloc[v].reset_index(drop=True), hold_df.iloc[te].reset_index(drop=True)
if len(train_df) > TRAIN_SAMPLE:
    train_df = train_df.sample(TRAIN_SAMPLE, random_state=SEED).reset_index(drop=True)
splits = {"train": train_df, "val": val_df, "test": test_df}
print("split sizes:", {k: len(v) for k, v in splits.items()})

norm = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
class DS(Dataset):
    def __init__(self, d, size): self.d=d; self.tf=T.Compose([T.Resize((size,size)),T.ToTensor(),norm]); self.size=size
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (self.size,self.size), (128,128,128))
        return self.tf(img)
def predict(model, d, size):
    loader = DataLoader(DS(d, size), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    P = []
    with torch.no_grad():
        for imgs in loader:
            with torch.amp.autocast('cuda'): p = torch.sigmoid(model(imgs.to(device, non_blocking=True)))
            P.append(p.float().cpu().numpy())
    return np.vstack(P)

out = {f"y_{sp}": d[TARGET_LABELS].to_numpy(dtype="float32") for sp, d in splits.items()}
for arch, (pat, size) in CKPTS.items():
    ck = best_ckpt(pat)
    model = (RadDino() if arch == "rad_dino" else CNNDualPool(arch)).to(device).eval()
    sd = torch.load(ck, map_location=device); sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"{arch}: loaded {ck.split('/')[-1]} (missing={len(miss)} unexpected={len(unexp)})")
    for sp, d in splits.items():
        out[f"p_{arch}_{sp}"] = predict(model, d, size)
    print(f"  {arch} predictions done")

np.savez_compressed("/workspace/chexpert_preds.npz", labels=np.array(TARGET_LABELS), **out)
print("\nSAVED /workspace/chexpert_preds.npz  — download this to your laptop")
