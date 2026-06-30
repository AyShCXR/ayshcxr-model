# eval_full_metrics.py
# AyShCXR — Professor's full metrics table (BME Lecture 2 format)
# by Subhrakant Sethi & Ayush Singh
#
# For a trained EfficientNet-B4 CheXpert checkpoint, computes:
#   Accuracy, Precision, Recall, F1, Sensitivity, Specificity  for Train / Val / Test
#   plus AUC-ROC (mean over diseases)
# Outputs one row of the professor's table, saved to metrics_table.csv
#
# Run after training:  python eval_full_metrics.py

import os, json, glob
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score

# ── Config (must match training) ───────────────────────────────────────────
PNG_ROOT      = "/chexpert_plus_png_412/PNG"
CHEXBERT_JSON = "/workspace/report_fixed.json"
MAIN_CSV      = "/workspace/df_chexpert_plus_240401.csv"
IMG_SIZE      = 380
BATCH         = 128
VAL_SPLIT     = 0.08
RANDOM_SEED   = 42
THRESHOLD     = 0.5   # probability threshold for binary metrics
MODEL_NAME    = "EfficientNet-B4 (CheXpert Plus)"

TARGET_LABELS = (
    "Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Find best checkpoint ────────────────────────────────────────────────────
ckpts = sorted(glob.glob("/workspace/efficientnet_chexpert_best_*.pth"))
if not ckpts:
    print("❌ No best checkpoint found (efficientnet_chexpert_best_*.pth). Train first.")
    raise SystemExit
CKPT = ckpts[-1]
print(f"Using checkpoint: {CKPT}")

# ── Data loading (same as training) ─────────────────────────────────────────
def load_data(split):
    records = {}
    with open(CHEXBERT_JSON) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            pk = r.get("path_to_image")
            if pk is None: continue
            records[pk] = {d: (1.0 if (r.get(d) is not None and float(r.get(d)) == 1.0) else 0.0)
                           for d in TARGET_LABELS}
    df = pd.read_csv(MAIN_CSV)
    if "frontal_lateral" in df.columns:
        df = df[df["frontal_lateral"] == "Frontal"]
    df = df[df["split"] == split].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        jpg = r["path_to_image"]
        png = os.path.join(PNG_ROOT, jpg.replace(".jpg", ".png"))
        if os.path.exists(png) and jpg in records:
            e = {"full_path": png}; e.update(records[jpg]); rows.append(e)
    return pd.DataFrame(rows)

tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class DS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["full_path"]).convert("RGB")
        return tf(img), torch.tensor(row[list(TARGET_LABELS)].to_numpy(dtype="float32"))

# build train / val / test splits
full_train = load_data("train")
from sklearn.model_selection import train_test_split
tr_df, va_df = train_test_split(full_train, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
te_df = load_data("valid")   # CheXpert official valid set = our Test (radiologist-labeled)
print(f"Train {len(tr_df)} | Val {len(va_df)} | Test {len(te_df)}")

# ── Model ───────────────────────────────────────────────────────────────────
def build_model():
    m = models.efficientnet_b4(weights=None)   # 3-channel (matches training)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 14))
    return m

model = build_model().to(device)
state = torch.load(CKPT, map_location=device, weights_only=False)
# strip torch.compile prefix if present
state = { k.replace("_orig_mod.", ""): v for k, v in state.items() }
model.load_state_dict(state, strict=False)
model.eval()
print("✅ Model loaded")

# ── Inference + metrics ─────────────────────────────────────────────────────
def infer(df):
    loader = DataLoader(DS(df), batch_size=BATCH, shuffle=False, num_workers=8)
    P, Y = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            with torch.amp.autocast('cuda'):
                p = torch.sigmoid(model(imgs.to(device)))
            P.append(p.float().cpu().numpy()); Y.append(labs.numpy())
    return np.vstack(P), np.vstack(Y)

def metrics(P, Y):
    pred = (P >= THRESHOLD).astype(int)
    # macro-average across the 14 diseases
    accs, precs, recs, f1s, specs, aucs = [], [], [], [], [], []
    for i in range(len(TARGET_LABELS)):
        y, ph, pr = Y[:, i], pred[:, i], P[:, i]
        tp = ((ph == 1) & (y == 1)).sum(); tn = ((ph == 0) & (y == 0)).sum()
        fp = ((ph == 1) & (y == 0)).sum(); fn = ((ph == 0) & (y == 1)).sum()
        acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)          # = sensitivity
        spec = tn / max(tn + fp, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        accs.append(acc); precs.append(prec); recs.append(rec)
        f1s.append(f1); specs.append(spec)
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, pr))
    return {
        "Acc": np.mean(accs), "Prec": np.mean(precs), "Recall": np.mean(recs),
        "F1": np.mean(f1s), "Sens": np.mean(recs), "Spec": np.mean(specs),
        "AUC": np.mean(aucs),
    }

print("\nRunning inference on Train / Val / Test ...")
splits = {"Train": tr_df, "Val": va_df, "Test": te_df}
results = {}
for name, df in splits.items():
    P, Y = infer(df)
    results[name] = metrics(P, Y)
    print(f"  {name} done")

# ── Build the professor's table row ─────────────────────────────────────────
row = {"Model": MODEL_NAME}
for sp in ["Train", "Val", "Test"]:
    m = results[sp]
    row[f"{sp} Acc"]    = round(m["Acc"], 4)
    row[f"{sp} Prec"]   = round(m["Prec"], 4)
    row[f"{sp} Recall"] = round(m["Recall"], 4)
    row[f"{sp} F1"]     = round(m["F1"], 4)
    row[f"{sp} Sens"]   = round(m["Sens"], 4)
    row[f"{sp} Spec"]   = round(m["Spec"], 4)
row["AUC-ROC"] = round(results["Test"]["AUC"], 4)

table = pd.DataFrame([row])
out = "/workspace/metrics_table.csv"
table.to_csv(out, index=False)
print(f"\n✅ Saved professor's table → {out}\n")
pd.set_option("display.width", 200, "display.max_columns", 50)
print(table.to_string(index=False))
