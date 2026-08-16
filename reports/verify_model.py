# verify_model.py
# AyShCXR — does the model actually detect disease? 2026-08-10
#
# Runs the FULL deployed pipeline (all loaded models -> merge -> calibrate ->
# salience) over radiologist-labelled VinDr images and reports, per finding:
#
#   AUC          can it rank a diseased film above a healthy one?
#   sensitivity  when the disease IS present, how often does it flag it?
#   specificity  when the disease is ABSENT, how often does it stay quiet?
#   top-1 / top-3  on films with exactly one finding, does it name it?
#
# Ground truth = >=2 of 3 radiologists agreed. This is the honest test: the
# models never saw VinDr in training, it comes from Vietnamese hospitals, and
# the labels were written by humans rather than by software.
#
#   python reports/verify_model.py              600 images
#   python reports/verify_model.py 1500         more images, slower
#   python reports/verify_model.py 600 --cases  also print example predictions

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(f"Cannot find .ayshcxr_root above {_HERE}")
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
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score

with contextlib.redirect_stdout(io.StringIO()):
    import model_loader as ml
    import calibration as cal
    import clinical_extras as cx
    import medical_knowledge_ext as ext
    from disease_ontology import MODEL_REGISTRY

MIN_POS = 20


def main():
    n_req = 600
    show_cases = "--cases" in _sys.argv
    for a in _sys.argv[1:]:
        if a.isdigit():
            n_req = int(a)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = [m for m, s in MODEL_REGISTRY.items() if s["verified"]]
    with contextlib.redirect_stdout(io.StringIO()):
        LOADED, _ = ml.load_models(ids, dev)

    TH = json.load(open("results/disease_thresholds.json"))["thresholds"]
    BASIS = json.load(open("results/disease_thresholds.json")).get("basis", {})
    BASE = json.load(open("results/finding_baselines.json"))["baselines"]

    df = pd.read_csv("data/vindr/vindr_labels.csv")
    LAB = [c for c in df.columns if c not in ("image_id", "full_path", "n_readers")]
    covered = ml.covered_findings(LOADED)
    testable = [c for c in LAB if c in covered and c != "No Finding"]

    print("=" * 78)
    print("  AyShCXR — MODEL VERIFICATION")
    print("=" * 78)
    print(f"\n  pipeline   : {len(LOADED)} models -> merge -> calibrate -> salience")
    print(f"  models     : {[e['id'] for e in LOADED]}")
    print(f"  test data  : VinDr-CXR, labelled by 3 radiologists each")
    print(f"               (never seen in training; Vietnamese hospitals)")
    print(f"  ground truth: >=2 of 3 radiologists agreed")
    print(f"  testable   : {len(testable)} of {len(LAB)} VinDr findings "
          f"({len(LAB)-len(testable)} not in any training set)")

    rng = np.random.RandomState(0)
    idx = set()
    for f in testable:
        pos = df.index[df[f] == 1].tolist()
        if pos:
            idx |= set(rng.choice(pos, min(90, len(pos)), replace=False).tolist())
    norm = df.index[df["No Finding"] == 1].tolist()
    idx |= set(rng.choice(norm, min(max(n_req - len(idx), 120), len(norm)),
                          replace=False).tolist())
    sub = df.loc[sorted(idx)].reset_index(drop=True)
    print(f"  images     : {len(sub)} (stratified so rare findings appear)")

    print(f"\n  running ...")
    t0 = time.time()
    rows, keep = [], []
    for i, r in sub.iterrows():
        try:
            img = Image.open(r["full_path"]).convert("RGB")
        except Exception:
            continue
        merged, _, _ = ml.predict(LOADED, img, dev, mc_passes=0)
        rows.append(cal.calibrate_all(merged)); keep.append(i)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(sub)}  ({time.time()-t0:.0f}s)")
    P = pd.DataFrame(rows)
    S = sub.loc[keep].reset_index(drop=True)
    print(f"  done in {time.time()-t0:.0f}s\n")

    # ── per-finding detection ───────────────────────────────────────────────
    print("=" * 78)
    print("  1. CAN IT DETECT EACH DISEASE?")
    print("=" * 78)
    print(f"\n{'finding':<24}{'n_pos':>6}{'AUC':>7}{'sens':>7}{'spec':>7}"
          f"{'thresh':>8}  verdict")
    print("-" * 78)
    good, weak, useless = [], [], []
    for f in testable:
        y = S[f].values
        if y.sum() < MIN_POS:
            print(f"{f:<24}{int(y.sum()):>6}      -      -      -        -  "
                  f"too few positives to judge")
            continue
        p = P[f].values
        auc = roc_auc_score(y, p)
        t = (TH.get(f) or {}).get("detected", 0.5)
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        star = "*" if BASIS.get(f) != "measured" else " "
        if auc >= 0.80:
            v = "STRONG — trust this"; good.append(f)
        elif auc >= 0.70:
            v = "usable, correlate clinically"; weak.append(f)
        elif auc >= 0.60:
            v = "WEAK — treat with caution"; weak.append(f)
        else:
            v = "NO SIGNAL — do not rely on"; useless.append(f)
        print(f"{f:<24}{int(y.sum()):>6}{auc:>7.3f}{sens:>7.0%}{spec:>7.0%}"
              f"{t:>7.2f}{star}  {v}")
    print("-" * 78)
    print("  * threshold from baseline distribution, not validated against labels")
    print("  AUC 0.50 = random guessing.  1.00 = perfect.")
    print("  sens = catches the disease when present."
          "   spec = stays quiet when absent.")

    # ── naming the right disease ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  2. ON A FILM WITH ONE FINDING, DOES IT NAME THAT FINDING?")
    print("=" * 78)
    S2 = S.copy(); S2["n"] = S2[testable].sum(axis=1)
    one = S2[(S2.n == 1) & (S2["No Finding"] == 0)]
    if len(one):
        ranks = []
        for i in one.index:
            sc = {c: P.loc[i, c] for c in testable if ext.is_pathology(c)}
            order = [d["disease"] for d in cx.rank_by_salience(
                [{"disease": k, "probability": v} for k, v in sc.items()], BASE)]
            truth = one.loc[i, testable].idxmax()
            ranks.append((truth, order.index(truth) + 1 if truth in order else None))
        RK = pd.DataFrame(ranks, columns=["truth", "rank"]).dropna()
        print(f"\n{'finding':<24}{'n':>5}{'named 1st':>11}{'in top 3':>10}")
        print("-" * 52)
        for d, g in RK.groupby("truth"):
            print(f"{d:<24}{len(g):>5}{(g['rank']==1).mean():>10.0%}"
                  f"{(g['rank']<=3).mean():>10.0%}")
        print("-" * 52)
        print(f"{'OVERALL':<24}{len(RK):>5}{(RK['rank']==1).mean():>10.0%}"
              f"{(RK['rank']<=3).mean():>10.0%}")
        print(f"\n  random guessing would name it 1st "
              f"{1/len([c for c in testable if ext.is_pathology(c)]):.0%} of the time")

    # ── normal films ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  3. ON A NORMAL FILM, DOES IT STAY QUIET?")
    print("=" * 78)
    nf = S[S["No Finding"] == 1]
    if len(nf):
        flags = []
        for i in nf.index:
            n = sum(1 for c in testable
                    if ext.is_pathology(c)
                    and P.loc[i, c] >= (TH.get(c) or {}).get("detected", 0.5))
            flags.append(n)
        flags = np.array(flags)
        print(f"\n  {len(nf)} radiologist-confirmed NORMAL films")
        print(f"  flagged nothing      : {(flags==0).mean():>6.0%}")
        print(f"  flagged 1 finding    : {(flags==1).mean():>6.0%}")
        print(f"  flagged 2+           : {(flags>=2).mean():>6.0%}")
        print(f"  average false flags  : {flags.mean():>6.2f} per normal film")
        print("\n  Every false flag is a patient sent for tests they do not need.")

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"\n  TRUST (AUC >= 0.80)      : {', '.join(good) if good else 'none'}")
    print(f"  CAUTION (0.60-0.80)      : {', '.join(weak) if weak else 'none'}")
    print(f"  DO NOT RELY (< 0.60)     : {', '.join(useless) if useless else 'none'}")
    print("\n  This is an external validation: different country, different")
    print("  hospitals, different machines, human labels. Numbers here are")
    print("  a fairer estimate of real performance than any in-domain test.")
    print("=" * 78)

    if show_cases:
        print("\n\n" + "=" * 78)
        print("  EXAMPLE PREDICTIONS — inspect these yourself")
        print("=" * 78)
        shown = 0
        for f in good[:3]:
            pos = S.index[S[f] == 1].tolist()[:2]
            for i in pos:
                sc = {c: P.loc[i, c] for c in testable if ext.is_pathology(c)}
                top = cx.rank_by_salience(
                    [{"disease": k, "probability": v} for k, v in sc.items()], BASE)[:4]
                truth = [c for c in testable if S.loc[i, c] == 1]
                print(f"\n  {_os.path.basename(S.loc[i,'full_path'])}")
                print(f"    radiologists said : {truth}")
                print(f"    model said        : " + ", ".join(
                    f"{t['disease']} {t['probability']:.2f}" for t in top))
                hit = top[0]["disease"] in truth
                print(f"    -> {'CORRECT (named it first)' if hit else 'named something else first'}")
                shown += 1
        print(f"\n  ({shown} examples shown — run with more images for more)")


if __name__ == "__main__":
    main()
