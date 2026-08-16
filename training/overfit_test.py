# overfit_test.py
# DEFINITIVE diagnostic: can the model+pipeline even FIT a tiny dataset?
# Trains on 200 images for 400 steps with high LR, no augmentation.
# A working CNN should drive TRAIN AUC to ~0.95+ (memorize 200 images easily).
#   - If TRAIN AUC -> 0.95+  : model/pipeline/optimization are FINE.
#                              The problem is purely generalization -> full run + more
#                              data + more epochs is justified.
#   - If TRAIN AUC stays low : there is a fundamental pipeline/optimization bug
#                              (and no amount of data/epochs will help) -> must debug.

import os, json, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score

PNG_ROOT      = "/chexpert_plus_png_412/PNG"
CHEXBERT_JSON = "/workspace/report_fixed.json"
MAIN_CSV      = "/workspace/df_chexpert_plus_240401.csv"
IMG_SIZE      = 380
N_IMAGES      = 200
N_STEPS       = 400
BATCH         = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_LABELS = (
    "Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding",
)

print("Loading labels...")
records = {}
with open(CHEXBERT_JSON) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        pk = r.get("path_to_image")
        if pk: records[pk] = {d: (1.0 if (r.get(d) is not None and float(r.get(d))==1.0) else 0.0) for d in TARGET_LABELS}

df = pd.read_csv(MAIN_CSV)
if "frontal_lateral" in df.columns: df = df[df["frontal_lateral"]=="Frontal"]
df = df[df["split"]=="train"].reset_index(drop=True)
rows = []
for _, r in df.iterrows():
    jpg = r["path_to_image"]; png = os.path.join(PNG_ROOT, jpg.replace(".jpg",".png"))
    if os.path.exists(png) and jpg in records:
        e={"full_path":png}; e.update(records[jpg]); rows.append(e)
    if len(rows) >= N_IMAGES: break
data = pd.DataFrame(rows)
print(f"Overfit set: {len(data)} images")

tf = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(s, df): s.df=df.reset_index(drop=True)
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        row=s.df.iloc[i]
        img=Image.open(row["full_path"]).convert("RGB")
        return tf(img), torch.tensor(row[list(TARGET_LABELS)].to_numpy(dtype="float32"))

loader = DataLoader(DS(data), batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)

model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
inf_ = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Linear(inf_, 14))
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # high LR to force memorization
crit = nn.BCEWithLogitsLoss()
model.train()

print("\n=== OVERFIT TEST (200 imgs, plain BCE, lr=1e-3, NO aug) ===")
it = iter(loader)
for step in range(1, N_STEPS+1):
    try: imgs, labs = next(it)
    except StopIteration: it = iter(loader); imgs, labs = next(it)
    imgs, labs = imgs.to(device), labs.to(device)
    opt.zero_grad()
    loss = crit(model(imgs), labs)
    loss.backward()
    opt.step()
    if step % 40 == 0 or step == 1:
        print(f"step {step:3d}  loss {loss.item():.4f}")

# train AUC on the SAME 200 images
model.eval(); P, Y = [], []
with torch.no_grad():
    for imgs, labs in DataLoader(DS(data), batch_size=BATCH, shuffle=False):
        P.append(torch.sigmoid(model(imgs.to(device))).cpu().numpy()); Y.append(labs.numpy())
P=np.vstack(P); Y=np.vstack(Y)
aucs=[roc_auc_score(Y[:,i],P[:,i]) for i in range(14) if len(np.unique(Y[:,i]))>1]
print(f"\nTRAIN AUC on the 200 images it memorized: {np.mean(aucs):.4f}")
print("VERDICT:")
print("  TRAIN AUC > 0.90  -> model/pipeline FINE; issue is generalization (run full).")
print("  TRAIN AUC < 0.70  -> FUNDAMENTAL pipeline/optimization bug; must debug, not train.")
