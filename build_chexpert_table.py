# build_chexpert_table.py  —  RUN ON YOUR LAPTOP (no GPU, no images needed)
# Builds the professor's BME-format metrics table from chexpert_preds.npz (+ cv_results.csv).
# Multi-label correct: per-disease metrics at a tuned threshold, then macro-averaged.
#
#   put chexpert_preds.npz (and cv_results.csv) in the same folder, then:
#   python build_chexpert_table.py
#
# needs: numpy pandas scikit-learn matplotlib   (pip install them if missing)

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc as auc_fn

PREDS = "chexpert_preds.npz"          # download this from the server
CV    = "cv_results.csv"              # download this when the CV finishes (optional)

d = np.load(PREDS, allow_pickle=True)
LABELS = list(d["labels"])
ARCHS  = {"efficientnet_b4": "EfficientNet-B4", "densenet121": "DenseNet-121", "rad_dino": "Rad-DINO"}
SPLITS = ["train", "val", "test"]

# probabilities per model per split + the ensemble (mean of the 3)
P = {sp: {a: d[f"p_{a}_{sp}"] for a in ARCHS} for sp in SPLITS}
for sp in SPLITS:
    P[sp]["ensemble"] = np.mean([P[sp][a] for a in ARCHS], axis=0)
Y = {sp: d[f"y_{sp}"] for sp in SPLITS}
MODELS = list(ARCHS.values()) + ["Ensemble"]
key = {**{ARCHS[a]: a for a in ARCHS}, "Ensemble": "ensemble"}

def tuned_thresholds(probs, y):                 # per-disease threshold that maximizes F1, tuned on VAL
    from sklearn.metrics import precision_recall_curve
    th = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        if len(np.unique(y[:, i])) < 2: continue
        prec, rec, t = precision_recall_curve(y[:, i], probs[:, i])
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if len(t) > 0:
            th[i] = float(t[int(np.argmax(f1[:len(t)]))])
    return th

def metrics(probs, y, th):                       # multi-label: per-disease then macro-avg
    n_d = y.shape[1]; acc=prec=rec=spec=f1=auc=0; cnt=0; cnt_auc=0
    accs=[]; precs=[]; recs=[]; specs=[]; f1s=[]; aucs=[]
    for i in range(n_d):
        pred = (probs[:, i] >= th[i]).astype(int); yt = y[:, i].astype(int)
        TP=int(((pred==1)&(yt==1)).sum()); FP=int(((pred==1)&(yt==0)).sum())
        TN=int(((pred==0)&(yt==0)).sum()); FN=int(((pred==0)&(yt==1)).sum())
        accs.append((TP+TN)/max(TP+TN+FP+FN,1))
        precs.append(TP/max(TP+FP,1)); recs.append(TP/max(TP+FN,1)); specs.append(TN/max(TN+FP,1))
        p_,r_=precs[-1],recs[-1]; f1s.append(2*p_*r_/max(p_+r_,1e-9))
        if len(np.unique(yt))>1: aucs.append(roc_auc_score(yt, probs[:, i]))
    return [np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(recs), np.mean(specs)], np.mean(aucs)
    # order: Acc, Prec, Recall, F1, Sensitivity(=Recall), Specificity

# 5 CheXpert "competition" diseases — cleaner labels, the official-leaderboard subset
COMP5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
comp5_idx = [LABELS.index(d) for d in COMP5]
def auc5(probs, y):
    return float(np.mean([roc_auc_score(y[:, i], probs[:, i]) for i in comp5_idx if len(np.unique(y[:, i])) > 1]))

# CV column (from cv_results.csv if present)
cv_map = {}
if os.path.exists(CV):
    cvdf = pd.read_csv(CV)
    cv_map = {ARCHS.get(r["arch"], r["arch"]): r["cv_mean_auc"] for _, r in cvdf.iterrows()}
    cv_map["Ensemble"] = np.mean(list(cv_map.values())) if cv_map else np.nan

rows = []
for m in MODELS:
    k = key[m]
    th = tuned_thresholds(P["val"][k], Y["val"])           # tune on val, apply to all splits
    tr, _   = metrics(P["train"][k], Y["train"], th)
    va, _   = metrics(P["val"][k],   Y["val"],   th)
    te, auc = metrics(P["test"][k],  Y["test"],  th)
    rows.append([m, *tr, *va, *te, cv_map.get(m, np.nan), auc, auc5(P["test"][k], Y["test"])])

cols = ["Model",
    "Train Acc","Train Prec","Train Recall","Train F1","Train Sens","Train Spec",
    "Val Acc","Val Prec","Val Recall","Val F1","Val Sens","Val Spec",
    "Test Acc","Test Prec","Test Recall","Test F1","Test Sens","Test Spec",
    "CV AUC","AUC-ROC","AUC (5-dx)"]
res = pd.DataFrame(rows, columns=cols).round(4)
res.to_csv("CheXpert_Model_Comparison.csv", index=False)
print(res.to_string(index=False))
print("\nsaved CheXpert_Model_Comparison.csv")

# styled table (for notebook display): res.style.background_gradient(cmap="YlGnBu")

# ROC curves (micro-average over the 14 diseases) on TEST
plt.figure(figsize=(7,7))
for m in MODELS:
    k = key[m]; yt = Y["test"].ravel(); pr = P["test"][k].ravel()
    fpr, tpr, _ = roc_curve(yt, pr)
    plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC={auc_fn(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("CheXpert Test ROC (micro-average)"); plt.legend(loc="lower right"); plt.grid(True)
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight"); print("saved roc_curves.png")

# AUC bar chart
plt.figure(figsize=(8,5))
bars = plt.bar(res["Model"], res["AUC-ROC"])
for b in bars: plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.4f}", ha="center", va="bottom")
plt.ylim(0.7, 0.9); plt.ylabel("AUC-ROC"); plt.title("CheXpert AUC by Model")
plt.savefig("auc_bar.png", dpi=150, bbox_inches="tight"); print("saved auc_bar.png")
