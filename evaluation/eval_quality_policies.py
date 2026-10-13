# eval_quality_policies.py
# AyShCXR — Which quality-conditioned abstention policy is sound, judged on
# held-out films?  (Follow-up to eval_quality_abstention.py.)
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS EXISTS
#   eval_quality_abstention.py showed quality-conditioned abstention catches far
#   more degradation-induced errors than uncertainty-only abstention. It also
#   showed a weakness: residual risk among the findings still reported ROSE
#   (0.231 -> 0.256), because the policy withholds every affected finding on a
#   faulty film, including the many correct negatives. This script tests whether
#   a better-founded policy fixes that, without tuning anything on test films.
#
# DESIGN (fixed before any result was seen)
#   1. GPU pass, run once: for every film, the clean image and 8 graded faults
#      are scored by the ensemble. Every raw value (label, clean prob, degraded
#      prob, uncertainty, quality severity) is cached to CSV.
#   2. Offline pass: policies are re-evaluated from the cache in seconds. None of
#      them has a fitted parameter; all thresholds are the ones already fixed in
#      clinical_extras.py / image_quality.py / disease_thresholds.json.
#   3. Policies (pre-declared):
#        none          report everything
#        unc           model uncertainty only (prior work's signal)
#        qual_all      quality: withhold affected findings (current policy)
#        qual_pos      quality: withhold only POSITIVE calls on affected findings
#                      ("cannot confirm on this film") — most induced errors are
#                      false positives the fault manufactured
#        comb_all      unc OR qual_all
#        comb_pos      unc OR qual_pos
#   4. Selection on half A of the films, report on held-out half B. Rule: among
#      policies whose residual risk <= no-abstention risk, pick the one with the
#      largest gain in induced errors caught over uncertainty-ranked abstention
#      at the same coverage. Ties -> higher coverage.
#   5. Clean films are scored too, so the cost of the method (abstaining on
#      good films) is measured, not assumed.
#   6. 95% CIs by bootstrap over films (resampling films, not decisions, because
#      the 14 findings of one film are not independent).
#
# USAGE
#   AYSHCXR_MODELS="chexpert_densenet121_v1,chexpert_efficientnet_b4_v1,chexpert_rad_dino_v1,nih_densenet121_v1" \
#     python evaluation/eval_quality_policies.py --n 300            # GPU pass + analysis
#   python evaluation/eval_quality_policies.py --analyze-only       # re-analyse the cache

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

import clinical_extras as cx

CACHE = "results/quality_policy_records.csv"
SUMMARY = "results/quality_policy_summary.csv"
THRESH_JSON = "results/disease_thresholds.json"
POLICIES = ["none", "unc", "qual_all", "qual_pos", "comb_all", "comb_pos"]


# ── 1. GPU pass ───────────────────────────────────────────────────────────────
def build_cache(n, passes, seed, out):
    import torch
    from PIL import Image
    import model_loader as ml
    import image_quality as iq
    from eval_quality_abstention import (degrade, infer, load_split,
                                         DEGRADATIONS, NIH14)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = [m.strip() for m in _os.environ.get(
        "AYSHCXR_MODELS", "nih_densenet121_v1").split(",") if m.strip()]
    loaded, failed = ml.load_models(ids, device)
    for mid, why in failed:
        print(f"  ! {mid} unavailable — {why}", file=_sys.stderr)
    if not loaded:
        _sys.exit("No models could be loaded.")

    df = load_split(n, seed)
    print(f"  GPU pass: {len(df)} films x (clean + {len(DEGRADATIONS)} faults) | "
          f"{passes} MC passes | device {device} | models {[e['id'] for e in loaded]}")
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        try:
            base = Image.open(r["rpath"]).convert("RGB")
        except Exception:
            continue
        clean = infer(loaded, base, device, passes)
        csev = iq.affected_severity(iq.assess(base))
        conds = [("clean", base, csev, clean)]
        for kind in DEGRADATIONS:
            d = degrade(base, kind)
            conds.append((kind, d, iq.affected_severity(iq.assess(d)),
                          infer(loaded, d, device, passes)))
        for kind, _img, sev, pred in conds:
            for f in NIH14:
                p, u = pred[f]
                rows.append({"film": r["Image Index"], "fault": kind, "finding": f,
                             "label": int(r[f]), "p_clean": clean[f][0],
                             "p": p, "u": u, "sev": sev.get(f) or ""})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  ...{i + 1}/{len(df)} films  ({el / 60:.1f} min, "
                  f"~{el / (i + 1) * (len(df) - i - 1) / 60:.0f} min left)")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  cached {len(rows)} decisions -> {out}  "
          f"({(time.time() - t0) / 60:.1f} min)")


# ── 1b. re-score quality only, with site-calibrated thresholds (CPU) ──────────
# Predictions do not depend on the quality thresholds, so the cached model
# outputs are reused; only `sev` is recomputed for every (film, condition).
def _requality_film(args):
    film, path, thr = args
    from PIL import Image
    import image_quality as iq
    from site_calibrate_quality import apply
    from eval_quality_abstention import degrade, DEGRADATIONS
    apply(thr)
    base = Image.open(path).convert("RGB")
    out = {}
    for kind in ["clean"] + DEGRADATIONS:
        img = base if kind == "clean" else degrade(base, kind)
        for f, s in iq.affected_severity(iq.assess(img)).items():
            out[(film, kind, f)] = s
    return out


def requality(cache, thr, out, workers):
    from multiprocessing import Pool
    rec = pd.read_csv(cache)
    lab = pd.read_csv("data/nih_full_labels.csv", usecols=["Image Index", "full_path"])
    lab["rpath"] = lab["full_path"].map(
        lambda p: str(p).lstrip(".").lstrip("\\/").replace("\\", "/"))
    paths = dict(zip(lab["Image Index"], lab["rpath"]))
    jobs = [(f, paths[f], thr) for f in rec["film"].unique()]
    sev = {}
    with Pool(workers) as pool:
        for d in pool.imap_unordered(_requality_film, jobs, chunksize=4):
            sev.update(d)
    rec["sev"] = [sev.get((a, b, c), "") for a, b, c in
                  zip(rec["film"], rec["fault"], rec["finding"])]
    rec.to_csv(out, index=False)
    print(f"  re-scored quality for {len(jobs)} films with {thr} -> {out}")


# ── 2. offline policy evaluation ──────────────────────────────────────────────
def add_decisions(rec):
    with open(THRESH_JSON, encoding="utf-8") as f:
        th = json.load(f).get("thresholds", {})
    t = rec["finding"].map(lambda f: (th.get(f) or {}).get("detected", 0.5))
    rec = rec.copy()
    rec["sev"] = rec["sev"].fillna("").astype(str)
    lab = rec["label"].astype(bool)
    pos = rec["p"] >= t
    rec["pos"] = pos
    rec["error"] = pos != lab
    clean_ok = (rec["p_clean"] >= t) == lab
    rec["induced"] = (rec["fault"] != "clean") & clean_ok & rec["error"]
    rec["induced_fp"] = rec["induced"] & pos & ~lab

    unc, qual = [], []
    for p, u, f, s in zip(rec["p"], rec["u"], rec["finding"], rec["sev"]):
        unc.append(cx.should_abstain(p, u, f)["abstain"])
        qual.append(cx.should_abstain(p, 0.0, f, quality_severity=(s or None))["abstain"])
    rec["ab_none"] = False
    rec["ab_unc"] = unc
    rec["ab_qual_all"] = qual
    rec["ab_qual_pos"] = rec["ab_qual_all"] & rec["pos"]
    rec["ab_comb_all"] = rec["ab_unc"] | rec["ab_qual_all"]
    rec["ab_comb_pos"] = rec["ab_unc"] | rec["ab_qual_pos"]
    return rec


def metrics(deg, clean, pol):
    """deg/clean: decision frames. Returns the numbers reported per policy."""
    ab = deg["ab_" + pol].to_numpy()
    kept = ~ab
    n, k = len(deg), int(ab.sum())
    risk = deg["error"].to_numpy()[kept].mean() if kept.any() else 0.0
    caught = int(deg["induced"].to_numpy()[ab].sum())
    # fair baselines at the SAME coverage (same number of abstentions k)
    order = np.argsort(-deg["u"].to_numpy(), kind="stable")
    ab_u = np.zeros(n, bool)
    ab_u[order[:k]] = True
    caught_u = int(deg["induced"].to_numpy()[ab_u].sum())
    risk_u = deg["error"].to_numpy()[~ab_u].mean() if (~ab_u).any() else 0.0
    return {
        "policy": pol,
        "coverage": kept.mean(),
        "risk": risk,
        "induced_total": int(deg["induced"].sum()),
        "induced_caught": caught,
        "induced_fp_caught": int(deg["induced_fp"].to_numpy()[ab].sum()),
        "unc_ranked_caught_same_cov": caught_u,
        "unc_ranked_risk_same_cov": risk_u,
        "gain_over_unc_ranked": caught - caught_u,
        "clean_abstain_rate": clean["ab_" + pol].mean() if len(clean) else np.nan,
        "clean_risk": (clean["error"].to_numpy()[~clean["ab_" + pol].to_numpy()].mean()
                       if len(clean) else np.nan),
    }


def table(rec, films):
    sub = rec[rec["film"].isin(films)]
    deg, clean = sub[sub["fault"] != "clean"], sub[sub["fault"] == "clean"]
    return pd.DataFrame([metrics(deg, clean, p) for p in POLICIES])


def select(tab):
    base = float(tab.loc[tab.policy == "none", "risk"].iloc[0])
    ok = tab[(tab.policy != "none") & (tab.risk <= base + 1e-12)]
    if ok.empty:
        return None
    return ok.sort_values(["gain_over_unc_ranked", "coverage"],
                          ascending=False).iloc[0]["policy"]


def bootstrap(rec, films, pol, reps, seed):
    rng = np.random.default_rng(seed)
    films = np.asarray(films)
    by = {f: g for f, g in rec[rec["film"].isin(films)].groupby("film")}
    out = []
    for _ in range(reps):
        pick = rng.choice(films, size=len(films), replace=True)
        sub = pd.concat([by[f] for f in pick if f in by], ignore_index=True)
        deg, clean = sub[sub["fault"] != "clean"], sub[sub["fault"] == "clean"]
        m = metrics(deg, clean, pol)
        b = metrics(deg, clean, "none")
        out.append((m["risk"] - b["risk"], m["gain_over_unc_ranked"],
                    m["induced_caught"] / max(m["induced_total"], 1)))
    a = np.array(out)
    return {name: (np.percentile(a[:, j], 2.5), np.percentile(a[:, j], 97.5))
            for j, name in enumerate(["risk_change", "gain_over_unc", "share_caught"])}


def analyze(cache, seed, reps):
    from threshold_overlap import threshold_tuning_films
    rec = pd.read_csv(cache)
    tuned = threshold_tuning_films()
    n_drop = rec.loc[rec["film"].isin(tuned), "film"].nunique()
    rec = add_decisions(rec[~rec["film"].isin(tuned)])
    print(f"  excluded {n_drop} film(s) that were used to tune the decision thresholds")
    films = np.array(sorted(rec["film"].unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(films)
    half_a, half_b = perm[: len(perm) // 2], perm[len(perm) // 2:]

    fmt = {"coverage": "{:.3f}", "risk": "{:.3f}", "unc_ranked_risk_same_cov": "{:.3f}",
           "clean_abstain_rate": "{:.3f}", "clean_risk": "{:.3f}"}

    def show(title, tab):
        print(f"\n  {title}")
        t = tab.copy()
        for c, f in fmt.items():
            t[c] = t[c].map(f.format)
        print(t[["policy", "coverage", "risk", "induced_caught", "induced_total",
                 "unc_ranked_caught_same_cov", "unc_ranked_risk_same_cov",
                 "gain_over_unc_ranked", "clean_abstain_rate"]].to_string(index=False))

    print("=" * 96)
    print(f"  QUALITY-CONDITIONED ABSTENTION: POLICY CHECK  "
          f"({len(films)} films; half A {len(half_a)} / half B {len(half_b)})")
    print("=" * 96)
    tab_a = table(rec, half_a)
    show("HALF A (selection only)", tab_a)
    chosen = select(tab_a)
    print(f"\n  selection rule on half A -> {chosen or 'NO policy keeps risk <= no-abstention'}")

    tab_b = table(rec, half_b)
    show("HALF B (held out; these are the numbers to report)", tab_b)
    tab_all = table(rec, films)
    show("ALL films (descriptive)", tab_all)

    # per-fault breakdown on half B for the chosen policy (or comb_pos if none)
    pol = chosen or "comb_pos"
    sub = rec[rec["film"].isin(half_b)]
    clean_b = sub[sub["fault"] == "clean"]
    print(f"\n  per-fault, half B, policy {pol}:")
    rows = []
    for k, g in sub[sub["fault"] != "clean"].groupby("fault", sort=False):
        m, b = metrics(g, clean_b, pol), metrics(g, clean_b, "none")
        rows.append({"fault": k, "induced": m["induced_total"], "caught": m["induced_caught"],
                     "unc_same_cov": m["unc_ranked_caught_same_cov"],
                     "risk_none": round(b["risk"], 3), "risk_pol": round(m["risk"], 3),
                     "coverage": round(m["coverage"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    ci = bootstrap(rec, half_b, pol, reps, seed)
    print(f"\n  95% bootstrap CI over half-B films ({reps} resamples), policy {pol}:")
    print(f"    residual-risk change vs none : [{ci['risk_change'][0]:+.4f}, {ci['risk_change'][1]:+.4f}]")
    print(f"    induced caught beyond unc    : [{ci['gain_over_unc'][0]:.0f}, {ci['gain_over_unc'][1]:.0f}]")
    print(f"    share of induced caught      : [{ci['share_caught'][0]:.3f}, {ci['share_caught'][1]:.3f}]")

    out = pd.concat([tab_a.assign(split="A"), tab_b.assign(split="B"),
                     tab_all.assign(split="ALL")], ignore_index=True)
    out["chosen_on_A"] = chosen or ""
    summary = (SUMMARY if _os.path.abspath(cache) == _os.path.abspath(CACHE)
               else _os.path.splitext(cache)[0] + "_summary.csv")
    out.to_csv(summary, index=False)
    print(f"\n  wrote {summary}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="Held-out check of quality-abstention policies")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--requality", default="",
                    help="site threshold JSON: re-score quality on the cached films "
                         "(CPU only) and analyse that variant")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.requality:
        site = a.cache.replace(".csv", "_site.csv")
        requality(a.cache, a.requality, site, a.workers)
        analyze(site, a.seed, a.boot)
        return
    if not a.analyze_only:
        build_cache(a.n, a.passes, a.seed, a.cache)
    analyze(a.cache, a.seed, a.boot)


if __name__ == "__main__":
    main()
