# verify_nih.py
# AyShCXR — test the deployed pipeline on NIH ChestX-ray14. 2026-08-10
#
# WHY: VinDr covers only ~10 findings and does not label Pneumonia or Edema at
# all — the two that matter most for a rural PHC. NIH has ground truth for all
# 14 of its findings, including 1,431 pneumonia and 2,303 edema cases, and the
# images are already on disk.
#
# HONEST CAVEAT ON THIS GROUND TRUTH
#   NIH labels were mined from radiology reports by an NLP tool, not written by
#   radiologists looking at the films. They are reported to be roughly 90%
#   accurate, and the "Pneumonia" label in particular is known to be noisy.
#   So this is a WEAKER test than the VinDr one — it measures agreement with an
#   automated labeller, not with a doctor. Treat the numbers as indicative.
#
#   python reports/verify_nih.py [n_per_finding]

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import io, json, time, warnings, contextlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score

with contextlib.redirect_stdout(io.StringIO()):
    import model_loader as ml, calibration as cal, clinical_extras as cx
    import medical_knowledge_ext as ext
    from disease_ontology import MODEL_REGISTRY

NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
       "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
       "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]


def main():
    n_per = int(_sys.argv[1]) if len(_sys.argv) > 1 else 80
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        LOADED, _ = ml.load_models(
            [m for m, s in MODEL_REGISTRY.items() if s["verified"]], dev)
    BASE = json.load(open("results/finding_baselines.json"))["baselines"]
    TH = json.load(open("results/disease_thresholds.json"))["thresholds"]

    df = pd.read_csv("data/nih_full_labels.csv")
    df["n_find"] = df[NIH].sum(axis=1)

    print("=" * 80)
    print("  NIH ChestX-ray14 verification — the 9 previously untested findings")
    print("=" * 80)
    print(f"\n  models     : {len(LOADED)}")
    print(f"  ground truth: NLP-mined from reports (~90% accurate), NOT")
    print(f"                radiologist-read. Weaker evidence than VinDr.")

    rng = np.random.RandomState(0)
    idx = set()
    for f in NIH:                                  # positives for each finding
        pos = df.index[df[f] == 1].tolist()
        if pos:
            idx |= set(rng.choice(pos, min(n_per, len(pos)), replace=False).tolist())
    healthy = df.index[df["n_find"] == 0].tolist()  # confirmed normal films
    idx |= set(rng.choice(healthy, min(250, len(healthy)), replace=False).tolist())
    sub = df.loc[sorted(idx)].reset_index(drop=True)
    print(f"  images     : {len(sub):,} (stratified; {min(250,len(healthy))} normals)")

    print("\n  running ...")
    t0 = time.time(); rows, keep = [], []
    for i, r in sub.iterrows():
        try:
            img = Image.open(r["full_path"]).convert("RGB")
        except Exception:
            continue
        m, _, _ = ml.predict(LOADED, img, dev, mc_passes=0)
        rows.append(cal.calibrate_all(m)); keep.append(i)
        if (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(sub)}  ({time.time()-t0:.0f}s)")
    P = pd.DataFrame(rows); S = sub.loc[keep].reset_index(drop=True)
    print(f"  done in {time.time()-t0/1:.0f}s\n")

    print("=" * 80)
    print("  DETECTION — can it tell a diseased film from a healthy one?")
    print("=" * 80)
    print(f"\n{'finding':<22}{'n_pos':>7}{'AUC':>8}{'sens':>7}{'spec':>7}  verdict")
    print("-" * 80)
    results = {}
    for f in NIH:
        y = S[f].values.astype(int)
        if y.sum() < 20 or len(np.unique(y)) < 2:
            print(f"{f:<22}{int(y.sum()):>7}       -      -      -  too few")
            continue
        p = P[f].values
        auc = roc_auc_score(y, p)
        t = (TH.get(f) or {}).get("detected", 0.5)
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        v = ("STRONG" if auc >= .80 else "usable" if auc >= .70
             else "WEAK" if auc >= .60 else "NO SIGNAL")
        results[f] = auc
        print(f"{f:<22}{int(y.sum()):>7}{auc:>8.3f}{sens:>7.0%}{spec:>7.0%}  {v}")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("  NAMING — on a film with exactly ONE finding, does it name it?")
    print("=" * 80)
    one = S[S[NIH].sum(axis=1) == 1]
    if len(one):
        rk = []
        for i in one.index:
            sc = {c: P.loc[i, c] for c in NIH if ext.is_pathology(c)}
            order = [d["disease"] for d in cx.rank_by_salience(
                [{"disease": k, "probability": v} for k, v in sc.items()], BASE)]
            truth = one.loc[i, NIH].idxmax()
            rk.append((truth, order.index(truth) + 1))
        RK = pd.DataFrame(rk, columns=["truth", "rank"])
        print(f"\n{'finding':<22}{'n':>5}{'named 1st':>11}{'top 3':>8}")
        print("-" * 48)
        for d, g in RK.groupby("truth"):
            print(f"{d:<22}{len(g):>5}{(g['rank']==1).mean():>10.0%}{(g['rank']<=3).mean():>8.0%}")
        print("-" * 48)
        print(f"{'OVERALL':<22}{len(RK):>5}{(RK['rank']==1).mean():>10.0%}"
              f"{(RK['rank']<=3).mean():>8.0%}   (random = {1/14:.0%})")

    print("\n" + "=" * 80)
    print("  NORMAL FILMS — does it stay quiet?")
    print("=" * 80)
    nf = S[S[NIH].sum(axis=1) == 0]
    if len(nf):
        flags = np.array([
            sum(1 for c in NIH if ext.is_pathology(c)
                and P.loc[i, c] >= (TH.get(c) or {}).get("detected", 0.5))
            for i in nf.index])
        print(f"\n  {len(nf)} films with NO finding reported")
        print(f"    flagged nothing : {(flags==0).mean():>6.0%}")
        print(f"    flagged 1       : {(flags==1).mean():>6.0%}")
        print(f"    flagged 2+      : {(flags>=2).mean():>6.0%}")
        print(f"    mean false flags: {flags.mean():>6.2f} per healthy film")

    print("\n" + "=" * 80)
    print("  THE NINE PREVIOUSLY UNTESTED FINDINGS")
    print("=" * 80)
    for f in ["Pneumonia", "Edema", "Mass", "Nodule", "Emphysema", "Hernia"]:
        a = results.get(f)
        print(f"    {f:<22}" + (f"AUC {a:.3f}" if a else "still untestable"))
    print("\n  (Enlarged Cardiomediastinum, Pleural Other and Fracture are")
    print("   CheXpert-only findings — NIH does not label them either.)")
    print("=" * 80)


if __name__ == "__main__":
    main()
