# eval_real_quality.py
# AyShCXR — Do REAL technical faults hurt the findings image_quality says they
# hurt?  (Real-film evidence for quality-conditioned abstention.)
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS EXISTS
#   eval_quality_abstention.py / eval_quality_policies.py use synthetic
#   degradations. This script uses no synthetic faults at all: it runs the
#   quality checker over real NIH test films, splits them by the faults it
#   actually detects, and asks whether the model's per-finding performance drops
#   on faulty films, and drops more for the findings AFFECTED_BY says a fault
#   corrupts than for the ones it does not.
#
# CONFOUNDING (stated, not hidden)
#   Technically poor films are more often AP / portable / ICU films, i.e. sicker
#   patients and different disease prevalence. Every comparison is therefore made
#   WITHIN view position (PA vs AP, from Data_Entry_2017.csv). AUC is
#   prevalence-independent; PPV is not, so PPV is reported with prevalence.
#   Bootstrap CIs resample PATIENTS (films of one patient are not independent).
#
# STAGES
#   --stage quality   CPU: assess every test film      -> results/real_quality_checks.csv
#   --stage predict   GPU: one ensemble pass per film  -> results/real_quality_preds.csv
#   --stage analyze   offline                          -> results/real_quality_summary.csv

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(f"Could not locate the AyShCXR project root above {_HERE}.")
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "evaluation")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import time
import numpy as np
import pandas as pd

LABELS_CSV = "data/nih_full_labels.csv"
META_CSV = "data/Data_Entry_2017.csv"
TEST_SPLIT = "data/test_list.txt"
THRESH_JSON = "results/disease_thresholds.json"
Q_OUT = "results/real_quality_checks.csv"
P_OUT = "results/real_quality_preds.csv"
S_OUT = "results/real_quality_summary.csv"
CHECKS = ["rotation", "exposure", "sharpness"]


def test_films():
    df = pd.read_csv(LABELS_CSV)
    meta = pd.read_csv(META_CSV)[["Image Index", "Patient ID", "View Position",
                                  "Patient Age"]]
    with open(TEST_SPLIT, encoding="utf-8") as f:
        test = {ln.strip() for ln in f if ln.strip()}
    df = df[df["Image Index"].isin(test)].drop(columns=["Patient ID"])
    df = df.merge(meta, on="Image Index", how="left")

    def resolve(p):
        rel = str(p).lstrip(".").lstrip("\\/").replace("\\", "/")
        return rel if _os.path.exists(rel) else None

    df["rpath"] = df["full_path"].map(resolve)
    return df[df["rpath"].notna()].reset_index(drop=True)


# ── stage 1: quality on every film (CPU, parallel) ────────────────────────────
def _assess_one(args):
    name, path, thr = args
    import image_quality as iq
    from PIL import Image
    if thr:                                     # site-calibrated thresholds
        from site_calibrate_quality import apply
        apply(thr)
    try:
        res = iq.assess(Image.open(path))
    except Exception as e:                      # unreadable file
        return {"Image Index": name, "available": False, "error": str(e)[:80]}
    row = {"Image Index": name, "available": res.get("available", False)}
    for c in res.get("checks", []):
        row[f"{c['check']}_sev"] = c["severity"]
        row[f"{c['check']}_val"] = c.get("value")
    return row


def stage_quality(workers, thr, q_out):
    from multiprocessing import Pool
    df = test_films()
    jobs = [(n, p, thr) for n, p in zip(df["Image Index"], df["rpath"])]
    print(f"  quality stage: {len(jobs)} real test films on {workers} CPU workers"
          f" | thresholds: {thr or 'image_quality.py defaults (VinDr)'}")
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_assess_one, jobs, chunksize=32), 1):
            rows.append(r)
            if i % 2000 == 0:
                el = time.time() - t0
                print(f"  ...{i}/{len(jobs)}  ({el / 60:.1f} min, "
                      f"~{el / i * (len(jobs) - i) / 60:.0f} min left)")
    out = pd.DataFrame(rows)
    out.to_csv(q_out, index=False)
    print(f"  wrote {q_out} ({len(out)} films, {(time.time() - t0) / 60:.1f} min)")
    for c in CHECKS:
        col = f"{c}_sev"
        if col in out:
            print(f"    {c:10s} {out[col].value_counts().to_dict()}")


# ── stage 2: one ensemble pass per film (GPU) ─────────────────────────────────
def stage_predict(max_clean, seed):
    import torch
    from PIL import Image
    import model_loader as ml
    import calibration as cal
    from disease_ontology import LABEL_SETS
    nih14 = LABEL_SETS["NIH14"]

    q = pd.read_csv(Q_OUT)
    df = test_films().merge(q, on="Image Index", how="inner")
    flagged = np.zeros(len(df), bool)
    for c in CHECKS:
        col = f"{c}_sev"
        if col in df:
            flagged |= df[col].isin(["warn", "severe"]).to_numpy()
    clean = df[~flagged]
    if len(clean) > max_clean:
        clean = clean.sample(max_clean, random_state=seed)
    todo = pd.concat([df[flagged], clean]).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = [m.strip() for m in _os.environ.get(
        "AYSHCXR_MODELS", "nih_densenet121_v1").split(",") if m.strip()]
    loaded, failed = ml.load_models(ids, device)
    for mid, why in failed:
        print(f"  ! {mid} unavailable — {why}", file=_sys.stderr)
    print(f"  predict stage: {int(flagged.sum())} flagged + {len(clean)} clean films | "
          f"device {device} | models {[e['id'] for e in loaded]}")

    done = set()
    if _os.path.exists(P_OUT):                   # resumable
        done = set(pd.read_csv(P_OUT, usecols=["Image Index"])["Image Index"])
    rows, t0, n_new = [], time.time(), 0
    todo = todo[~todo["Image Index"].isin(done)]
    for i, r in enumerate(todo.itertuples(index=False), 1):
        try:
            img = Image.open(r.rpath).convert("RGB")
        except Exception:
            continue
        merged, _pm, unc = ml.predict(loaded, img, device, mc_passes=0)
        merged = cal.calibrate_all(merged)
        row = {"Image Index": r[todo.columns.get_loc("Image Index")]}
        for f in nih14:
            row[f"p_{f}"] = float(merged.get(f, np.nan))
            row[f"u_{f}"] = float(unc.get(f, np.nan))
        rows.append(row)
        n_new += 1
        if i % 500 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(P_OUT, mode="a", index=False,
                                      header=not _os.path.exists(P_OUT))
            rows = []
            el = time.time() - t0
            print(f"  ...{i}/{len(todo)}  ({el / 60:.1f} min, "
                  f"~{el / i * (len(todo) - i) / 60:.0f} min left)")
    print(f"  {n_new} new films predicted -> {P_OUT}")


# ── stage 3: analysis ─────────────────────────────────────────────────────────
def _patient_resamples(pids, reps, rng):
    """`reps` patient-level bootstrap resamples, each an array of row indices.
    Patients are drawn with replacement and all of a patient's films come along."""
    pids = np.asarray(pids)
    order = np.argsort(pids, kind="stable")
    _, start, count = np.unique(pids[order], return_index=True, return_counts=True)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(start), len(start))
        st, ln = start[pick], count[pick]
        offs = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
        out.append(order[np.repeat(st, ln) + offs])
    return out


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:      # too few to be meaningful
        return np.nan
    return roc_auc_score(y, s)


def stage_analyze(boot, seed, q_in, s_out):
    import image_quality as iq
    from disease_ontology import LABEL_SETS
    nih14 = LABEL_SETS["NIH14"]
    with open(THRESH_JSON, encoding="utf-8") as f:
        th = json.load(f).get("thresholds", {})

    from threshold_overlap import threshold_tuning_films
    print(f"  quality flags from: {q_in}")
    df = (test_films().merge(pd.read_csv(q_in), on="Image Index")
          .merge(pd.read_csv(P_OUT), on="Image Index"))
    tuned = threshold_tuning_films()
    print(f"  excluded {int(df['Image Index'].isin(tuned).sum())} film(s) used to "
          f"tune the decision thresholds")
    df = df[~df["Image Index"].isin(tuned)].reset_index(drop=True)
    flag_cols = [f"{c}_sev" for c in CHECKS if f"{c}_sev" in df]
    df["any_flag"] = df[flag_cols].isin(["warn", "severe"]).any(axis=1)
    rng = np.random.default_rng(seed)

    print("=" * 100)
    print(f"  REAL-FILM QUALITY CHECK — {len(df)} NIH test films "
          f"({int(df.any_flag.sum())} flagged, {int((~df.any_flag).sum())} clean)")
    print("  view mix:", df.groupby(["View Position", "any_flag"]).size().to_dict())
    print("=" * 100)

    rows = []
    for c in CHECKS:
        col = f"{c}_sev"
        if col not in df:
            continue
        for view in ("PA", "AP"):
            v = df[df["View Position"] == view]
            bad = v[v[col].isin(["warn", "severe"])]
            ref = v[~v["any_flag"]]                 # clean on EVERY check
            if len(bad) < 30:
                continue
            # one set of patient resamples per (check, view), shared by all findings
            rs_b = _patient_resamples(bad["Patient ID"], boot, rng) if boot else []
            rs_r = _patient_resamples(ref["Patient ID"], boot, rng) if boot else []
            for f in nih14:
                y_b, s_b = bad[f].to_numpy(), bad[f"p_{f}"].to_numpy()
                y_r, s_r = ref[f].to_numpy(), ref[f"p_{f}"].to_numpy()
                a_b, a_r = _auc(y_b, s_b), _auc(y_r, s_r)
                t = (th.get(f) or {}).get("detected", 0.5)
                pos_b, pos_r = s_b >= t, s_r >= t
                ppv_b = y_b[pos_b].mean() if pos_b.sum() >= 5 else np.nan
                ppv_r = y_r[pos_r].mean() if pos_r.sum() >= 5 else np.nan
                # bootstrap the AUC drop over patients
                lo = hi = np.nan
                if boot and np.isfinite(a_b) and np.isfinite(a_r):
                    ds = []
                    for ib, ir in zip(rs_b, rs_r):
                        d = _auc(y_r[ir], s_r[ir]) - _auc(y_b[ib], s_b[ib])
                        if np.isfinite(d):
                            ds.append(d)
                    if len(ds) > boot * 0.8:
                        lo, hi = np.percentile(ds, [2.5, 97.5])
                rows.append({
                    "check": c, "view": view, "finding": f,
                    "in_affected_by": f in iq.AFFECTED_BY.get(c, []),
                    "n_flagged": len(bad), "n_clean": len(ref),
                    "prev_flagged": y_b.mean(), "prev_clean": y_r.mean(),
                    "auc_flagged": a_b, "auc_clean": a_r,
                    "auc_drop": (a_r - a_b) if np.isfinite(a_b) and np.isfinite(a_r) else np.nan,
                    "drop_ci_lo": lo, "drop_ci_hi": hi,
                    "ppv_flagged": ppv_b, "ppv_clean": ppv_r,
                })
    res = pd.DataFrame(rows)
    res.to_csv(s_out, index=False)

    # the test of the map: do affected findings drop more than unaffected ones?
    print("\n  mean AUC drop (clean - flagged), affected vs unaffected findings:")
    g = (res.dropna(subset=["auc_drop"])
         .groupby(["check", "view", "in_affected_by"])["auc_drop"]
         .agg(["mean", "count"]).round(4))
    print(g.to_string())
    sig = res[(res.drop_ci_lo > 0)]
    print(f"\n  findings with a significant real-film AUC drop (95% CI > 0): {len(sig)}")
    if len(sig):
        print(sig[["check", "view", "finding", "in_affected_by", "auc_clean",
                   "auc_flagged", "auc_drop", "drop_ci_lo", "drop_ci_hi",
                   "n_flagged"]].round(3).to_string(index=False))
    print(f"\n  wrote {s_out}")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description="Real-film evidence for quality-conditioned abstention")
    ap.add_argument("--stage", choices=["quality", "predict", "analyze"], required=True)
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 2) - 2))
    ap.add_argument("--max-clean", type=int, default=1_000_000,
                    help="cap on unflagged films to predict (default: all)")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default="",
                    help="site threshold JSON from site_calibrate_quality.py")
    a = ap.parse_args()
    # site-calibrated runs read/write their own files so the two never mix
    tag = "_site" if a.thresholds else ""
    q_file = Q_OUT.replace(".csv", f"{tag}.csv")
    s_file = S_OUT.replace(".csv", f"{tag}.csv")
    if a.stage == "quality":
        stage_quality(a.workers, a.thresholds, q_file)
    elif a.stage == "predict":
        stage_predict(a.max_clean, a.seed)
    else:
        stage_analyze(a.boot, a.seed, q_file, s_file)


if __name__ == "__main__":
    main()
