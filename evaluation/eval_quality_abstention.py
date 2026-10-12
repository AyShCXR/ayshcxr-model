# eval_quality_abstention.py
# AyShCXR — Does quality-conditioned abstention catch errors that model
# uncertainty does not?  (The contribution ablation for the paper.)
# by Subhrakant Sethi & Ayush Singh
#
# THE QUESTION
#   Prior selective-prediction work on chest X-rays abstains on MODEL UNCERTAINTY
#   (entropy / MC-Dropout spread / inter-model disagreement). None of it looks at
#   the PHYSICAL QUALITY of the film. A model can be confidently wrong on a
#   technically bad radiograph — a rotated film that fakes cardiomegaly, an
#   over-penetrated film that hides nodules — and uncertainty-based abstention
#   never fires, because each model is individually "sure".
#
# THE EXPERIMENT
#   Take clean NIH test films the model reads CORRECTLY. Apply realistic technique
#   faults (over-/under-penetration, blur, rotation). Each fault can turn a
#   correct reading into a wrong one ("induced error"). We then compare two
#   abstention policies on the degraded films:
#       uncertainty-only : clinical_extras.should_abstain(p, u, finding)
#       quality-conditioned : ... + quality_severity from image_quality
#   and report, per fault and overall:
#       * induced errors, and how many each policy catches
#       * risk–coverage: error rate among the findings each policy still reports
#   The claim holds if quality-conditioned abstention removes materially more
#   induced errors (lower risk) than uncertainty-only, especially induced FALSE
#   POSITIVES — disease the fault manufactured.
#
# HONEST NOTES
#   * Degradations are synthetic. They are a controlled stand-in for real bad
#     films; the quality thresholds themselves were calibrated on 400 REAL VinDr
#     radiographs (see image_quality.py), so the detector is not tuned to these.
#   * A fault the quality module cannot detect (e.g. mild rotation) will NOT be
#     caught by quality-conditioned abstention — that limitation is reported, not
#     hidden.
#   * Single-model runs have no inter-model disagreement, so the uncertainty-only
#     baseline uses MC-Dropout spread alone. Pass the full ensemble via
#     AYSHCXR_MODELS for the strongest baseline.
#
# USAGE
#   python evaluation/eval_quality_abstention.py                 # 200 films, 10 MC passes
#   python evaluation/eval_quality_abstention.py --n 40 --passes 5   # quick local check
#   AYSHCXR_MODELS="chexpert_densenet121_v1,chexpert_efficientnet_b4_v1,nih_densenet121_v1" \
#       python evaluation/eval_quality_abstention.py             # full ensemble baseline

# ── AyShCXR path bootstrap (same pattern as the other evaluation scripts) ─────
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(f"Could not locate the AyShCXR project root above {_HERE}.")
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# ── end bootstrap ─────────────────────────────────────────────────────────────

import argparse
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

import model_loader as ml
import image_quality as iq
import clinical_extras as cx
import calibration as cal
from disease_ontology import LABEL_SETS

NIH14 = LABEL_SETS["NIH14"]          # the findings with NIH ground-truth labels
LABELS_CSV = "data/nih_full_labels.csv"
TEST_SPLIT = "data/test_list.txt"
THRESH_JSON = "results/disease_thresholds.json"


# ── realistic technique faults (operate on a grayscale uint8 array) ───────────
def _deg_over_penetrated(a):            # too dark — hides nodules/lesions
    return np.clip(a * 0.30, 0, 255).astype(np.uint8)


def _deg_under_penetrated(a):           # too bright — fakes infiltration
    return np.clip(255 - (255 - a) * 0.35, 0, 255).astype(np.uint8)


def _as_pil(a):
    return Image.fromarray(a).convert("RGB")


def degrade(img_pil, kind):
    """Return a degraded RGB copy, at graded severity so we can separate the
    'targeted fault' regime (a specific finding corrupted while the model stays
    confident) from the 'global degradation' regime (whole image ruined, model
    uncertain anyway). Exposure acts on intensity; blur/rotation use PIL so they
    are genuine optical/geometric corruptions."""
    if kind == "blur_mild":
        return img_pil.filter(ImageFilter.GaussianBlur(radius=2)).convert("RGB")
    if kind == "blur":
        return img_pil.filter(ImageFilter.GaussianBlur(radius=4)).convert("RGB")
    if kind == "rotate_mild":
        return img_pil.rotate(6, resample=Image.BILINEAR, fillcolor=0).convert("RGB")
    if kind == "rotate":
        return img_pil.rotate(12, resample=Image.BILINEAR, fillcolor=0).convert("RGB")
    a = np.asarray(img_pil.convert("L"), dtype=np.float32)
    if kind == "over_penetrated_mild":
        return _as_pil(np.clip(a * 0.55, 0, 255).astype(np.uint8))
    if kind == "over_penetrated":
        return _as_pil(_deg_over_penetrated(a))
    if kind == "under_penetrated_mild":
        return _as_pil(np.clip(255 - (255 - a) * 0.60, 0, 255).astype(np.uint8))
    if kind == "under_penetrated":
        return _as_pil(_deg_under_penetrated(a))
    raise ValueError(f"unknown degradation {kind!r}")


DEGRADATIONS = ["over_penetrated_mild", "over_penetrated",
                "under_penetrated_mild", "under_penetrated",
                "blur_mild", "blur", "rotate_mild", "rotate"]


# ── one inference → {finding: (prob, uncertainty)} for the NIH findings ───────
# Calibration is applied to match the deployed pipeline exactly (predict_single
# and app.py both calibrate before thresholding); skipping it would inflate the
# positive rate and the error counts.
def infer(loaded, img_pil, device, passes):
    merged, _per, unc = ml.predict(loaded, img_pil, device, mc_passes=passes)
    merged = cal.calibrate_all(merged)
    return {f: (float(merged.get(f, 0.0)), float(unc.get(f, 0.0))) for f in NIH14}


def load_split(n, seed):
    df = pd.read_csv(LABELS_CSV)
    with open(TEST_SPLIT, encoding="utf-8") as f:
        test = {ln.strip() for ln in f if ln.strip()}
    df = df[df["Image Index"].isin(test)].reset_index(drop=True)

    def resolve(p):
        rel = str(p).lstrip(".").lstrip("\\/").replace("\\", "/")
        return rel if _os.path.exists(rel) else None

    df["rpath"] = df["full_path"].map(resolve)
    df = df[df["rpath"].notna()].reset_index(drop=True)
    if len(df) > n:
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return df


# ── three policies + a matched-coverage comparison ───────────────────────────
# Each record is one (film, finding) decision on a degraded film, carrying the
# abstain verdict of THREE policies:
#     unc_only  — model uncertainty only (prior work's signal)
#     qual_only — physical film quality only (this work's signal)
#     combined  — abstain if EITHER fires (the deployed decision)
# We report each policy at its natural operating point (coverage / induced
# errors caught / residual risk). Because the three abstain different numbers of
# decisions, we also give the fair head-to-head: at the COMBINED policy's
# coverage, how many induced errors does uncertainty-ranked abstention catch?
# If combined > that, quality adds value beyond uncertainty at equal budget.
def _policy(recs, flag):
    ab = [r for r in recs if r[flag]]
    ans = [r for r in recs if not r[flag]]
    caught = sum(1 for r in ab if r["induced"])
    risk = (sum(1 for r in ans if r["error"]) / len(ans)) if ans else 0.0
    cov = len(ans) / (len(recs) or 1)
    return {"cov": cov, "caught": caught, "risk": risk, "k": len(ab)}


def summarize(name, recs):
    n = len(recs) or 1
    err = sum(1 for r in recs if r["error"])
    induced = sum(1 for r in recs if r["induced"])
    induced_fp = sum(1 for r in recs if r["induced_fp"])

    u = _policy(recs, "unc_only")
    q = _policy(recs, "qual_only")
    c = _policy(recs, "combined")

    # fair head-to-head: uncertainty-ranked abstention at the COMBINED budget
    ranked = sorted(recs, key=lambda r: -r["unc"])
    caught_u_matched = sum(1 for r in ranked[:c["k"]] if r["induced"])

    return {
        "fault": name, "decisions": len(recs),
        "induced_err": induced, "induced_fp": induced_fp, "risk_none": err / n,
        "cov_unc": u["cov"], "caught_unc": u["caught"], "risk_unc": u["risk"],
        "cov_qual": q["cov"], "caught_qual": q["caught"], "risk_qual": q["risk"],
        "cov_comb": c["cov"], "caught_comb": c["caught"], "risk_comb": c["risk"],
        "caught_unc_matched": caught_u_matched,     # at combined's coverage
    }


def main():
    ap = argparse.ArgumentParser(description="Quality-conditioned vs uncertainty-only abstention")
    ap.add_argument("--n", type=int, default=200, help="number of test films")
    ap.add_argument("--passes", type=int, default=10, help="MC-Dropout passes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/quality_abstention_results.csv")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = [m.strip() for m in _os.environ.get(
        "AYSHCXR_MODELS", "nih_densenet121_v1").split(",") if m.strip()]
    loaded, failed = ml.load_models(ids, device)
    for mid, why in failed:
        print(f"  ! {mid} unavailable — {why}", file=_sys.stderr)
    if not loaded:
        _sys.exit("No models could be loaded.")

    with open(THRESH_JSON, encoding="utf-8") as f:
        thresholds = json.load(f).get("thresholds", {})

    def thr(finding):
        return (thresholds.get(finding) or {}).get("detected", 0.5)

    df = load_split(a.n, a.seed)
    print("=" * 74)
    print("  QUALITY-CONDITIONED vs UNCERTAINTY-ONLY ABSTENTION")
    print(f"  device {device} | models {[e['id'] for e in loaded]}")
    print(f"  {len(df)} NIH test films | {a.passes} MC passes | faults {DEGRADATIONS}")
    print("=" * 74)

    records = {k: [] for k in DEGRADATIONS}

    for i, row in df.iterrows():
        try:
            base = Image.open(row["rpath"]).convert("RGB")
        except Exception:
            continue
        clean = infer(loaded, base, device, a.passes)

        for kind in DEGRADATIONS:
            dimg = degrade(base, kind)
            qsev = iq.affected_severity(iq.assess(dimg))
            dpred = infer(loaded, dimg, device, a.passes)
            for f in NIH14:
                label = int(row[f])
                cp, _ = clean[f]
                dp, du = dpred[f]
                clean_ok = (cp >= thr(f)) == bool(label)
                deg_pos = dp >= thr(f)
                deg_ok = deg_pos == bool(label)
                induced = clean_ok and not deg_ok
                sev = qsev.get(f)
                unc_only = cx.should_abstain(dp, du, f)["abstain"]
                # uncertainty forced to 0 isolates the quality-only branches
                qual_only = cx.should_abstain(dp, 0.0, f,
                                              quality_severity=sev)["abstain"]
                combined = cx.should_abstain(dp, du, f,
                                             quality_severity=sev)["abstain"]
                records[kind].append({
                    "error": not deg_ok,
                    "induced": induced,
                    "induced_fp": induced and deg_pos and label == 0,
                    "unc": du,
                    "unc_only": unc_only,
                    "qual_only": qual_only,
                    "combined": combined,
                })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(df)} films")

    all_recs = [r for k in DEGRADATIONS for r in records[k]]
    rows = [summarize(k, records[k]) for k in DEGRADATIONS] + [summarize("ALL", all_recs)]
    pd.DataFrame(rows).to_csv(a.out, index=False)

    print(f"\n  induced errors caught, by policy, at each policy's natural coverage")
    print(f"\n{'fault':<20}{'ind.err':>8}  "
          f"{'unc(cov)':>13}{'qual(cov)':>14}{'comb(cov)':>14}")
    print("-" * 74)
    for r in rows:
        print(f"{r['fault']:<20}{r['induced_err']:>8}  "
              f"{r['caught_unc']:>5}({r['cov_unc']:.2f})"
              f"{r['caught_qual']:>8}({r['cov_qual']:.2f})"
              f"{r['caught_comb']:>8}({r['cov_comb']:.2f})")
    print("-" * 74)
    o = rows[-1]
    print(f"\n  Induced errors: {o['induced_err']}  (false positives: {o['induced_fp']})")
    print(f"  Natural operating points (induced caught / coverage / residual risk):")
    print(f"    uncertainty-only : {o['caught_unc']:>4}  cov {o['cov_unc']:.2f}  "
          f"risk {o['risk_unc']:.3f}")
    print(f"    quality-only     : {o['caught_qual']:>4}  cov {o['cov_qual']:.2f}  "
          f"risk {o['risk_qual']:.3f}")
    print(f"    COMBINED         : {o['caught_comb']:>4}  cov {o['cov_comb']:.2f}  "
          f"risk {o['risk_comb']:.3f}")
    print(f"    (no abstention baseline risk: {o['risk_none']:.3f})")
    print(f"\n  FAIR head-to-head at the COMBINED policy's coverage {o['cov_comb']:.2f}:")
    print(f"    combined catches           : {o['caught_comb']}")
    print(f"    uncertainty-ranked catches : {o['caught_unc_matched']}  "
          f"(+{o['caught_comb'] - o['caught_unc_matched']} from adding quality)")
    print(f"\n  wrote {a.out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
