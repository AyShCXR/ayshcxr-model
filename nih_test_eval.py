# nih_test_eval.py
# Evaluates the trained NIH EfficientNet (3-channel + GMP+GAP) on the OFFICIAL
# locked test set (test_list.txt — images never seen in training).
#
#   python nih_test_eval.py --ckpt efficientnet_best_auc0.7853_ep60.pth

import argparse, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score

LABELS_CSV = "/workspace/nih_full_labels.csv"
TEST_LIST  = "/workspace/test_list.txt"
IMG_SIZE   = 380
TARGET_LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
                 "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
                 "Fibrosis","Pleural Thickening","Hernia"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAPGMP(nn.Module):                       # must match training model
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1); self.gmp = nn.AdaptiveMaxPool2d(1)
    def forward(self, x): return self.gap(x) + self.gmp(x)

def build_model():
    m = models.efficientnet_b4(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4),
        nn.Linear(in_f, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 14))
    m.avgpool = GAPGMP()
    return m

class DS(Dataset):
    def __init__(self, df, tf): self.df=df.reset_index(drop=True); self.tf=tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128,128,128))
        return self.tf(img), torch.tensor(r[TARGET_LABELS].to_numpy(dtype="float32"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
args = ap.parse_args()

df = pd.read_csv(LABELS_CSV)
with open(TEST_LIST) as f:
    test_imgs = set(l.strip() for l in f)
df = df[df["Image Index"].isin(test_imgs)].reset_index(drop=True)
print(f"locked test images: {len(df):,}")

tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
loader = DataLoader(DS(df, tf), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

model = build_model().to(device).eval()
sd = torch.load(args.ckpt, map_location=device)
if isinstance(sd, dict):
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}")

P, Y = [], []
with torch.no_grad():
    for imgs, labs in loader:
        with torch.amp.autocast('cuda'):
            p = torch.sigmoid(model(imgs.to(device)))
        P.append(p.float().cpu().numpy()); Y.append(labs.numpy())
P, Y = np.vstack(P), np.vstack(Y)

print("\n=== NIH LOCKED TEST AUC ===")
rows = []
for i, d in enumerate(TARGET_LABELS):
    if len(np.unique(Y[:, i])) > 1:
        a = roc_auc_score(Y[:, i], P[:, i]); rows.append((d, a)); print(f"  {d:<22} {a:.4f}")
mean = float(np.mean([a for _, a in rows]))
print(f"\nMEAN TEST AUC: {mean:.4f}   (radiologist benchmark 0.778)")
pd.DataFrame(rows, columns=["disease","test_auc"]).to_csv("/workspace/nih_test_per_disease.csv", index=False)
print("saved nih_test_per_disease.csv")
