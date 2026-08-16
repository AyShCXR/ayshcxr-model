# tune_thresholds.py
# AyShCXR — set thresholds by SPECIFICITY, not F1. 2026-08-10
#
# THE PROBLEM WITH F1-OPTIMAL THRESHOLDS
#   compute_thresholds.py maximised F1 per finding, which favours sensitivity.
#   Measured consequence: false-positive rates on healthy films of 36%
#   (Atelectasis), 27% (Edema), 21% (Cardiomegaly). Across twelve findings those
#   stack up — 0 of 250 confirmed-healthy chests came back clean, averaging 2.49
#   false diagnoses each.
#
#   For a screening tool that must be able to say "this film is normal", that is
#   the wrong trade. A false positive sends a patient on a bus to a district
#   hospital for nothing; in a rural PHC that costs a day's wages.
#
# WHAT THIS DOES
#   Sets each finding's threshold at a target SPECIFICITY measured on films with
#   no reported finding, then reports the sensitivity that buys. You choose the
#   operating point with the numbers in front of you rather than inheriting one.
#
#   python reports/tune_thresholds.py [--spec 0.95] [--write]

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

import io, json, time, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
import torch

with contextlib.redirect_stdout(io.StringIO()):
    import model_loader as ml, calibration as cal, finding_reliability as rel
    from disease_ontology import MODEL_REGISTRY, CANONICAL_ORDER

NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
       "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
       "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]

# ── PER-FINDING OPERATING POINTS ───────────────────────────────────────────
# One global specificity target is the wrong tool. The cost of MISSING a
# finding differs enormously between diseases, so the threshold should too.
#
# Lower specificity target = lower threshold = catches more, false-alarms more.
#
#   0.85  MISSING THIS CAN KILL. Accept false alarms to avoid missing it.
#         Pneumothorax  — tension pneumothorax is minutes-to-hours fatal
#         Pneumonia     — leading infectious killer in rural India, treatable
#
#   0.90  Needs prompt action; a false alarm costs a referral, a miss costs more.
#         Effusion      — may need drainage
#         Consolidation — pneumonia's radiographic sign
#         Edema         — can indicate acute heart failure
#         Mass, Nodule  — possible malignancy; late diagnosis is the main
#                         cause of poor lung-cancer outcomes
#
#   0.97  Chronic or incidental. A false alarm sends a well patient on a
#         three-hour bus ride; a miss is usually caught at the next visit.
#         Cardiomegaly, Emphysema, Atelectasis, Hernia, Infiltration
#
# These are clinical judgements about which error hurts the patient more, not
# statistical optima. They should be reviewed by a clinician before deployment.
SPEC_TARGET = {
    "Pneumothorax":  0.85,
    "Pneumonia":     0.85,
    "Effusion":      0.90,
    "Consolidation": 0.90,
    "Edema":         0.90,
    "Mass":          0.90,
    "Nodule":        0.90,
    "Cardiomegaly":  0.97,
    "Emphysema":     0.97,
    "Atelectasis":   0.97,
    "Infiltration":  0.97,

    #   0.995 VERY RARE AND NOT URGENT — specificity dominates.
    #         Hernia occurs in 227 of 112,120 NIH images: 0.2%. At 4% false
    #         positives, screening 1,000 patients yields ~1.4 true hernias
    #         against ~40 false alarms — 29 of every 30 alerts wrong, which is
    #         worse than useless: it trains staff to ignore the system. At 0.995
    #         the false alarms nearly vanish and precision rises to ~50%, for a
    #         sensitivity cost of about 13 points on a finding that is usually
    #         incidental and rarely urgent.
    #
    #         NOTE the contrast with Pneumonia, which is also uncommon (1.3%)
    #         but kept at HIGH sensitivity — because missing pneumonia in a
    #         rural PHC kills, and missing a hiatus hernia does not. Rarity
    #         alone does not set the threshold; rarity WEIGHTED BY HARM does.
    "Hernia":        0.995,
}
DEFAULT_SPEC = 0.95


def score(loaded, rows, dev):
    out = []
    for _, r in rows.iterrows():
        try:
            img = Image.open(r["full_path"]).convert("RGB")
        except Exception:
            continue
        m, _, _ = ml.predict(loaded, img, dev, mc_passes=0)
        out.append(cal.calibrate_all(m))
    return pd.DataFrame(out)


def main():
    target = 0.95
    write = "--write" in _sys.argv
    for i, a in enumerate(_sys.argv):
        if a == "--spec" and i + 1 < len(_sys.argv):
            target = float(_sys.argv[i + 1])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        loaded, _ = ml.load_models(
            [m for m, s in MODEL_REGISTRY.items() if s["verified"]], dev)

    df = pd.read_csv("data/nih_full_labels.csv")
    df["n"] = df[NIH].sum(axis=1)
    healthy = df[df.n == 0].sample(300, random_state=5)

    print("=" * 76)
    print(f"  Threshold tuning — target specificity {target:.0%}")
    print("=" * 76)
    print(f"\n  scoring {len(healthy)} healthy films ...")
    t0 = time.time()
    H = score(loaded, healthy, dev)
    print(f"  done ({time.time()-t0:.0f}s)\n")

    old = json.load(open("results/disease_thresholds.json"))
    new_th = dict(old["thresholds"])
    rows = []

    per_finding = "--global" not in _sys.argv
    for f in NIH:
        if not rel.can_diagnose(f):
            continue
        spec = SPEC_TARGET.get(f, DEFAULT_SPEC) if per_finding else target
        pos = df[df[f] == 1].sample(min(70, int((df[f] == 1).sum())), random_state=5)
        P = score(loaded, pos, dev)
        t_new = float(np.percentile(H[f], spec * 100))
        t_old = old["thresholds"].get(f, {}).get("detected", 0.5)
        rows.append({
            "finding": f, "spec": spec,
            "t_old": t_old, "t_new": round(t_new, 4),
            "fpr_old": float((H[f] >= t_old).mean()),
            "fpr_new": float((H[f] >= t_new).mean()),
            "sens_old": float((P[f] >= t_old).mean()),
            "sens_new": float((P[f] >= t_new).mean()),
        })
        new_th[f] = {"detected": round(t_new, 4),
                     "borderline": round(float(np.percentile(H[f], max(spec-0.15, 0.5)*100)), 4),
                     "spec_target": spec}

    R = pd.DataFrame(rows).sort_values("spec")
    print(f"{'finding':<20}{'spec':>6}{'threshold':>16}{'false pos':>16}{'sensitivity':>18}")
    print(f"{'':<20}{'goal':>6}{'old':>8}{'new':>8}{'old':>8}{'new':>8}{'old':>9}{'new':>9}")
    print("-" * 76)
    for _, r in R.iterrows():
        print(f"{r['finding']:<20}{r['spec']:>6.2f}{r['t_old']:>8.3f}{r['t_new']:>8.3f}"
              f"{r['fpr_old']:>8.0%}{r['fpr_new']:>8.0%}"
              f"{r['sens_old']:>9.0%}{r['sens_new']:>9.0%}")
    print("-" * 76)
    print(f"{'MEAN':<20}{'':>16}{R.fpr_old.mean():>8.0%}{R.fpr_new.mean():>8.0%}"
          f"{R.sens_old.mean():>9.0%}{R.sens_new.mean():>9.0%}")

    exp_old = R.fpr_old.sum()
    exp_new = R.fpr_new.sum()
    print(f"\n  expected false diagnoses on a healthy film:")
    print(f"    before {exp_old:.2f}    after {exp_new:.2f}")
    print(f"  average sensitivity: {R.sens_old.mean():.0%} -> {R.sens_new.mean():.0%}")
    print("\n  THE TRADE: fewer healthy patients sent for needless tests,")
    print("  at the cost of missing more true findings. For a PHC screening")
    print("  tool that must be able to report 'normal', this is the right")
    print("  direction — but it IS a real cost and should be stated in any")
    print("  claim about the system.")

    if write:
        old["thresholds"] = new_th
        old["note"] = (f"detected = {target:.0%}-specificity point measured on "
                       f"{len(healthy)} healthy NIH films; borderline = 80th "
                       f"percentile. Replaces F1-optimal thresholds, which gave "
                       f"{exp_old:.1f} false diagnoses per healthy film.")
        old["target_specificity"] = target
        json.dump(old, open("results/disease_thresholds.json", "w"), indent=1)
        print(f"\n  WROTE results/disease_thresholds.json")
    else:
        print(f"\n  (dry run — re-run with --write to apply)")


if __name__ == "__main__":
    main()
