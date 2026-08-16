# cross_dataset_eval.py
# Cross-dataset generalization: CheXpert-trained models evaluated on the NIH OFFICIAL
# test set, scored on the 7 diseases COMMON to both datasets. The 4 NIH-only diseases
# (Emphysema, Fibrosis, Pleural Thickening, Hernia) are excluded by construction.
#   python cross_dataset_eval.py

import glob, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score

NIH_CSV   = "/workspace/nih_full_labels.csv"
TEST_LIST = "/workspace/test_list.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CheXpert model output order (14)
CHEX = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
# common disease:  (display name, CheXpert output index, NIH column name)
COMMON = [("Cardiomegaly",1,"Cardiomegaly"), ("Edema",4,"Edema"), ("Consolidation",5,"Consolidation"),
          ("Pneumonia",6,"Pneumonia"), ("Atelectasis",7,"Atelectasis"), ("Pneumothorax",8,"Pneumothorax"),
          ("Pleural Effusion",9,"Effusion")]   # CheXpert "Pleural Effusion" == NIH "Effusion"

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
        self.gap = nn.AdaptiveAvgPool2d(1); self.gmp = nn.AdaptiveMaxPool2d(1); self.classifier = complex_head(in_f, n)
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

df = pd.read_csv(NIH_CSV)
with open(TEST_LIST) as f:
    test_imgs = set(l.strip() for l in f)
df = df[df["Image Index"].isin(test_imgs)].reset_index(drop=True)
print(f"NIH official test images: {len(df):,}")

norm = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
class DS(Dataset):
    def __init__(self, size): self.size=size; self.tf=T.Compose([T.Resize((size,size)),T.ToTensor(),norm])
    def __len__(self): return len(df)
    def __getitem__(self, i):
        r = df.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (self.size,self.size), (128,128,128))
        return self.tf(img)
def predict(arch, ckpt, size):
    m = (RadDino() if arch == "rad_dino" else CNNDualPool(arch)).to(device).eval()
    sd = torch.load(ckpt, map_location=device); sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    loader = DataLoader(DS(size), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    P = []
    with torch.no_grad():
        for imgs in loader:
            with torch.amp.autocast('cuda'): p = torch.sigmoid(m(imgs.to(device, non_blocking=True)))
            P.append(p.float().cpu().numpy())
    return np.vstack(P)

probs = {}
for arch, (pat, size) in CKPTS.items():
    ck = best_ckpt(pat)
    probs[arch] = predict(arch, ck, size)
    print(f"{arch} predictions done ({ck.split('/')[-1]})")

def report(name, P):
    print(f"\n=== {name}  (CheXpert -> NIH, 7 common diseases) ===")
    aucs = []
    for disp, ci, nih_col in COMMON:
        y = df[nih_col].to_numpy()
        if len(np.unique(y)) > 1:
            a = roc_auc_score(y, P[:, ci]); aucs.append(a); print(f"  {disp:<18} {a:.4f}")
    m = float(np.mean(aucs)); print(f"  MEAN (7 common): {m:.4f}")
    return m

rows = []
for arch in probs:
    rows.append((arch, report(arch, probs[arch])))
ens = np.mean(list(probs.values()), axis=0)
rows.append(("ensemble", report("ENSEMBLE", ens)))
pd.DataFrame(rows, columns=["model","crossdataset_mean_auc"]).to_csv("/workspace/crossdataset_results.csv", index=False)
print("\nsaved crossdataset_results.csv")
