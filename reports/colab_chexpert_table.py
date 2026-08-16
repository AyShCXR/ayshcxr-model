# colab_chexpert_table.py
# ── HOW TO USE IN GOOGLE COLAB ───────────────────────────────────────────────
# 1. Open a new Colab notebook (colab.research.google.com)
# 2. Copy this ENTIRE file into one code cell
# 3. Run it (Shift+Enter)
# 4. When prompted, upload BOTH:  chexpert_preds.npz  AND  cv_results.csv
# It then shows the colored table, ROC curves, and AUC bar chart inline.
# (Colab already has numpy/pandas/sklearn/matplotlib — no install needed.)
# ─────────────────────────────────────────────────────────────────────────────


# ── AyShCXR path bootstrap (added 2026-08-08 during folder reorganisation) ──
# This script now lives in a subfolder but still refers to data files by bare
# name (e.g. "nih_full_labels.csv"). Pointing the working directory at the
# project root keeps every existing relative path working unchanged, and puts
# core/ on sys.path so `import medical_knowledge` still resolves.
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(
        "Could not locate the AyShCXR project root: no .ayshcxr_root marker "
        f"found above {_HERE}. Restore that file or run from the project root."
    )
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ── end path bootstrap ──────────────────────────────────────────────────────

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc as auc_fn, precision_recall_curve
from IPython.display import display

# ── upload the two files ─────────────────────────────────────────────────────
try:
    from google.colab import files
    print(">>> Upload BOTH files now:  chexpert_preds.npz  and  cv_results.csv")
    files.upload()
except Exception:
    print("(not in Colab — reading chexpert_preds.npz / cv_results.csv from current folder)")

PREDS, CV = "chexpert_preds.npz", "cv_results.csv"

d = np.load(PREDS, allow_pickle=True)
LABELS = list(d["labels"])
ARCHS  = {"efficientnet_b4": "EfficientNet-B4", "densenet121": "DenseNet-121", "rad_dino": "Rad-DINO"}
SPLITS = ["train", "val", "test"]

P = {sp: {a: d[f"p_{a}_{sp}"] for a in ARCHS} for sp in SPLITS}
for sp in SPLITS:
    P[sp]["ensemble"] = np.mean([P[sp][a] for a in ARCHS], axis=0)
Y = {sp: d[f"y_{sp}"] for sp in SPLITS}
MODELS = list(ARCHS.values()) + ["Ensemble"]
key = {**{ARCHS[a]: a for a in ARCHS}, "Ensemble": "ensemble"}

def tuned_thresholds(probs, y):                 # per-disease threshold maximizing F1, tuned on VAL
    th = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        if len(np.unique(y[:, i])) < 2: continue
        prec, rec, t = precision_recall_curve(y[:, i], probs[:, i])
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if len(t) > 0:
            th[i] = float(t[int(np.argmax(f1[:len(t)]))])
    return th

def metrics(probs, y, th):                       # multi-label: per-disease then macro-avg
    accs=[]; precs=[]; recs=[]; specs=[]; f1s=[]; aucs=[]
    for i in range(y.shape[1]):
        pred = (probs[:, i] >= th[i]).astype(int); yt = y[:, i].astype(int)
        TP=int(((pred==1)&(yt==1)).sum()); FP=int(((pred==1)&(yt==0)).sum())
        TN=int(((pred==0)&(yt==0)).sum()); FN=int(((pred==0)&(yt==1)).sum())
        accs.append((TP+TN)/max(TP+TN+FP+FN,1))
        precs.append(TP/max(TP+FP,1)); recs.append(TP/max(TP+FN,1)); specs.append(TN/max(TN+FP,1))
        p_,r_=precs[-1],recs[-1]; f1s.append(2*p_*r_/max(p_+r_,1e-9))
        if len(np.unique(yt))>1: aucs.append(roc_auc_score(yt, probs[:, i]))
    return [np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(recs), np.mean(specs)], np.mean(aucs)

COMP5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
comp5_idx = [LABELS.index(x) for x in COMP5]
def auc5(probs, y):
    return float(np.mean([roc_auc_score(y[:, i], probs[:, i]) for i in comp5_idx if len(np.unique(y[:, i])) > 1]))

cv_map = {}
if os.path.exists(CV):
    cvdf = pd.read_csv(CV)
    cv_map = {ARCHS.get(r["arch"], r["arch"]): r["cv_mean_auc"] for _, r in cvdf.iterrows()}
    cv_map["Ensemble"] = np.mean(list(cv_map.values())) if cv_map else np.nan
else:
    print("(cv_results.csv not uploaded — CV AUC column will be blank)")

rows = []
for m in MODELS:
    k = key[m]
    th = tuned_thresholds(P["val"][k], Y["val"])
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

# ── colored styled table (the professor's look) ─────────────────────────────
display(res.set_index("Model").style.background_gradient(cmap="YlGnBu").format("{:.4f}"))

# ── ROC curves (micro-average over the 14 diseases) on TEST ─────────────────
plt.figure(figsize=(7,7))
for m in MODELS:
    k = key[m]; yt = Y["test"].ravel(); pr = P["test"][k].ravel()
    fpr, tpr, _ = roc_curve(yt, pr)
    plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC={auc_fn(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("CheXpert Test ROC (micro-average)"); plt.legend(loc="lower right"); plt.grid(True)
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight"); plt.show()

# ── AUC bar chart ───────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
bars = plt.bar(res["Model"], res["AUC-ROC"])
for b in bars: plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.4f}", ha="center", va="bottom")
plt.ylim(0.7, 0.9); plt.ylabel("AUC-ROC"); plt.title("CheXpert AUC by Model")
plt.savefig("auc_bar.png", dpi=150, bbox_inches="tight"); plt.show()

print("\nSaved: CheXpert_Model_Comparison.csv, roc_curves.png, auc_bar.png")
