# clinical_extras.py
# AyShCXR — anatomical zone localisation + uncertainty-based abstention.
# by Subhrakant Sethi & Ayush Singh — 2026-08-10
#
# Two additions that need no new model and no retraining:
#
# 1. ZONE LOCALISATION
#    Radiologists report WHERE, not just WHAT. "Consolidation, right lower
#    zone" carries different clinical meaning from "Consolidation, right upper
#    zone" — upper-zone disease raises tuberculosis, lower-zone raises
#    pneumonia. The app already produces a Grad-CAM heatmap; this simply reads
#    which of the six standard zones the attention falls in.
#
# 2. ABSTENTION
#    MC-Dropout uncertainty is already computed and displayed but nothing acts
#    on it. A model that answers confidently on an image it cannot read is more
#    dangerous than one that says "refer". This converts the number into a
#    decision.
#
# ── LIMITATION, STATED PLAINLY ──────────────────────────────────────────────
#   Zones are GEOMETRIC, not anatomical. The image is divided into thirds by
#   height and halves by width. True zones follow the lung fields, which vary
#   with body habitus, rotation and diaphragm position. Proper zoning needs a
#   lung segmentation model. This is a reasonable approximation for a frontal
#   film that is roughly centred, and it is reported as "approximate".
#
# Self-test:  python core/clinical_extras.py

import numpy as np

# ── 1. ZONES ────────────────────────────────────────────────────────────────
# NOTE ON LATERALITY: a chest radiograph is viewed as if facing the patient, so
# the patient's RIGHT lung appears on the LEFT of the image. Getting this
# backwards would put every finding on the wrong side of the body — a serious
# reporting error. The mapping below is deliberately explicit.
ZONE_ROWS = ("upper", "mid", "lower")
ZONE_COLS = ("right", "left")        # image-left = patient-RIGHT


def zone_of(heatmap, threshold=0.55):
    """Which anatomical zone does the model's attention fall in?

    heatmap: 2-D array, any scale. Returns dict with the dominant zone plus the
    full distribution, or None if the heatmap is unusable.
    """
    if heatmap is None:
        return None
    h = np.asarray(heatmap, dtype=np.float64)
    if h.ndim == 3:
        h = h.mean(axis=2)
    if h.ndim != 2 or h.size == 0:
        return None

    rng = h.max() - h.min()
    if rng < 1e-8:
        return None                      # flat heatmap carries no location
    h = (h - h.min()) / rng

    rows, cols = h.shape
    r1, r2 = rows // 3, 2 * rows // 3
    c1 = cols // 2

    blocks = {
        ("upper", "right"): h[:r1, :c1],   ("upper", "left"): h[:r1, c1:],
        ("mid",   "right"): h[r1:r2, :c1], ("mid",   "left"): h[r1:r2, c1:],
        ("lower", "right"): h[r2:, :c1],   ("lower", "left"): h[r2:, c1:],
    }

    # mass of ABOVE-THRESHOLD attention, not raw mean: a large lukewarm region
    # should not outvote a small intense focus
    mass = {k: float((v * (v >= threshold)).sum()) for k, v in blocks.items()}
    total = sum(mass.values())
    if total <= 0:
        return None

    dist = {f"{r} {c}": mass[(r, c)] / total for (r, c) in mass}
    top = max(mass, key=mass.get)
    share = mass[top] / total

    if share >= 0.55:
        conf = "focal"
    elif share >= 0.35:
        conf = "predominant"
    else:
        conf = "diffuse"

    return {
        "zone": f"{top[0]} zone, {top[1]} side" if conf != "diffuse"
                else "diffuse / multiple zones",
        "row": top[0], "side": top[1],
        "share": round(share, 3),
        "pattern": conf,
        "distribution": {k: round(v, 3) for k, v in
                         sorted(dist.items(), key=lambda kv: -kv[1])},
        "note": "Approximate — geometric thirds, not segmented lung fields. "
                "Image-left corresponds to the patient's right side.",
    }


def zone_hint(zone_info, disease):
    """Clinical significance of location, where it genuinely differs."""
    if not zone_info or zone_info["pattern"] == "diffuse":
        return None
    row = zone_info["row"]
    hints = {
        ("upper", "Consolidation"): "Upper-zone consolidation raises suspicion of tuberculosis — send sputum for AFB.",
        ("upper", "Infiltration"): "Upper-zone infiltration is characteristic of tuberculosis in endemic areas.",
        ("upper", "Lung Lesion"): "Upper-zone cavitating lesions suggest tuberculosis; exclude before assuming malignancy.",
        ("upper", "Fibrosis"): "Upper-zone fibrosis suggests old tuberculosis or silicosis.",
        ("lower", "Consolidation"): "Lower-zone consolidation is the typical distribution of bacterial pneumonia.",
        ("lower", "Effusion"): "Fluid collects in the lower zones under gravity — consistent with effusion.",
        ("lower", "Fibrosis"): "Lower-zone fibrosis suggests interstitial lung disease rather than old TB.",
        ("lower", "Atelectasis"): "Basal atelectasis is common post-operatively and with shallow breathing.",
    }
    return hints.get((row, disease))


# ── 2. ABSTENTION ───────────────────────────────────────────────────────────
# MC-Dropout standard deviation thresholds. Deliberately conservative: the cost
# of an unnecessary referral is a wasted trip; the cost of a confident wrong
# answer is a missed diagnosis.
UNCERTAIN_STD   = 0.10
VERY_UNCERTAIN  = 0.16
BORDERLINE_LO   = 0.35      # probability band where the model is least decisive
BORDERLINE_HI   = 0.65


def should_abstain(probability, uncertainty, disease=None):
    """Decide whether the system should decline to commit.

    Two independent triggers:
      * high spread across MC-Dropout passes — the model disagrees with itself
      * probability sitting in the indecisive middle band with elevated spread
    """
    p = float(probability)
    u = float(uncertainty or 0.0)

    # NaN compares False against every threshold below, so without this check
    # a broken output would fall through to "answer" — the least safe outcome.
    if not (np.isfinite(p) and np.isfinite(u)):
        return {"abstain": True, "level": "high",
                "reason": "Model output was not a valid number, so no "
                          "conclusion can be drawn from it.",
                "action": "Refer for radiologist review — do not rely on this result."}

    if u >= VERY_UNCERTAIN:
        return {"abstain": True, "level": "high",
                "reason": f"Model output varied widely across repeated passes "
                          f"(±{u*100:.1f}%). It cannot reach a stable conclusion "
                          f"on this image.",
                "action": "Refer for radiologist review — do not rely on this result."}

    if u >= UNCERTAIN_STD and BORDERLINE_LO <= p <= BORDERLINE_HI:
        return {"abstain": True, "level": "moderate",
                "reason": f"Result is borderline ({p*100:.0f}%) and unstable "
                          f"(±{u*100:.1f}%).",
                "action": "Clinical correlation required; consider referral or "
                          "repeat imaging."}

    return {"abstain": False, "level": "none", "reason": None, "action": None}


def panel_abstention(predictions):
    """Whole-study verdict.

    If the top findings are all uncertain, the STUDY is unreadable by the
    system, not merely one finding — that is a different and more important
    message for the health worker.
    """
    tops = sorted(predictions, key=lambda p: -p.get("probability", 0))[:3]
    flags = [should_abstain(p.get("probability", 0), p.get("uncertainty", 0),
                            p.get("disease")) for p in tops]
    high = sum(1 for f in flags if f["abstain"] and f["level"] == "high")
    any_ab = sum(1 for f in flags if f["abstain"])

    if high >= 2 or (tops and any_ab == len(tops)):
        return {"abstain": True, "level": "study",
                "message": "The system could not reach a confident conclusion "
                           "on this radiograph. Refer for radiologist review.",
                "per_finding": flags}
    return {"abstain": False, "level": "none", "message": None,
            "per_finding": flags}


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("  clinical_extras self-test")
    print("=" * 72)

    H = W = 120
    yy, xx = np.mgrid[0:H, 0:W]

    def blob(cy, cx, s=14):
        return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s ** 2)))

    print("\nZONE LOCALISATION")
    print("  (image-left = patient RIGHT)")
    cases = [
        ("attention top-left of image",     blob(H * 0.18, W * 0.25)),
        ("attention bottom-right of image", blob(H * 0.82, W * 0.75)),
        ("attention centre",                blob(H * 0.50, W * 0.50)),
        ("attention everywhere",            np.ones((H, W)) * 0.8),
    ]
    for name, hm in cases:
        z = zone_of(hm)
        if z:
            print(f"  {name:34s} -> {z['zone']:28s} "
                  f"{z['pattern']:11s} share {z['share']}")
        else:
            print(f"  {name:34s} -> no localisation (flat heatmap)")

    z = zone_of(blob(H * 0.18, W * 0.25))
    print(f"\n  clinical hint, upper-zone Consolidation:")
    print(f"    {zone_hint(z, 'Consolidation')}")
    z2 = zone_of(blob(H * 0.82, W * 0.75))
    print(f"  clinical hint, lower-zone Consolidation:")
    print(f"    {zone_hint(z2, 'Consolidation')}")

    print("\nABSTENTION")
    for p, u, label in [(0.91, 0.02, "confident positive"),
                        (0.08, 0.01, "confident negative"),
                        (0.52, 0.12, "borderline + unstable"),
                        (0.73, 0.19, "high spread"),
                        (0.45, 0.04, "borderline but stable")]:
        r = should_abstain(p, u)
        verdict = f"ABSTAIN ({r['level']})" if r["abstain"] else "answer"
        print(f"  p={p:.2f} ±{u:.2f}  {label:24s} -> {verdict}")

    print("\nWHOLE-STUDY VERDICT")
    unreadable = [{"disease": "Pneumonia", "probability": 0.55, "uncertainty": 0.18},
                  {"disease": "Consolidation", "probability": 0.50, "uncertainty": 0.17},
                  {"disease": "Effusion", "probability": 0.48, "uncertainty": 0.15}]
    readable = [{"disease": "Cardiomegaly", "probability": 0.93, "uncertainty": 0.02},
                {"disease": "Effusion", "probability": 0.20, "uncertainty": 0.03},
                {"disease": "Edema", "probability": 0.15, "uncertainty": 0.02}]
    for name, preds in (("unreadable study", unreadable), ("clear study", readable)):
        v = panel_abstention(preds)
        print(f"  {name:18s} -> abstain={v['abstain']}  {v['message'] or ''}")

    assert panel_abstention(unreadable)["abstain"] is True
    assert panel_abstention(readable)["abstain"] is False
    assert zone_of(np.ones((10, 10))) is None
    assert zone_of(None) is None

    print("\n" + "=" * 72)
    print("  ALL CHECKS PASSED")
    print("=" * 72)


# ── 3. SALIENCE RANKING ─────────────────────────────────────────────────────
# THE PROBLEM
#   Raw probability is not comparable across findings. Measured on 500 real
#   radiographs, the merged+calibrated pipeline produces these TYPICAL scores:
#       Infiltration        0.527      Cardiomegaly    0.074
#       Nodule              0.513      Pneumonia       0.072
#       Mass                0.488      Consolidation   0.057
#       Pleural Thickening  0.441      Edema           0.050
#   So Infiltration at 0.53 is completely ordinary, while Cardiomegaly at 0.35 is
#   nearly five times its usual value. Ranking by raw probability puts the
#   ordinary finding first every time: measured top-1 accuracy for Cardiomegaly
#   was 10% despite an AUC of 0.913. The model SEES the disease; the ranking
#   buries it.
#
# THE FIX
#   Rank by  salience = probability / that finding's own detection threshold.
#   Thresholds come from reports/compute_thresholds.py, fitted against
#   radiologist consensus, so this is grounded in labelled data rather than an
#   arbitrary rescaling.
#
#       salience > 1  -> above the level that indicates disease for THIS finding
#       salience = 1  -> exactly at threshold
#       salience < 1  -> below it, however large the raw number looks
#
#   The DISPLAYED probability is unchanged — this only affects ordering and
#   which findings are surfaced. Showing a clinician a rescaled number would be
#   worse than showing an honest one in a sensible order.

# WHICH NORMALISATION — measured, not assumed.
#   Three variants were A/B tested on 333 VinDr films where >=2 of 3
#   radiologists agreed on exactly one finding:
#
#                        top-1   top-3   Effusion  Pneumothorax  Cardiomegaly
#     raw probability     30%     63%       84%        82%            9%
#     p / threshold       33%     71%        9%(!)     73%           67%
#     z vs baseline       39%     64%       91%        91%           62%
#
#   Dividing by the detection threshold looked good in aggregate but destroyed
#   Effusion (84% -> 9%), which is both the strongest finding (AUC 0.935) and one
#   of the most clinically urgent. Aggregate top-1 treats every finding as
#   equally important; clinically they are not.
#
#   The z-score against each finding's own baseline spread won on top-1 AND
#   improved the two most important findings. It works because (p90 - median)
#   measures EFFECT SIZE: how much this finding's score actually moves when the
#   disease is present. Effusion jumps hard on a true effusion so it ranks high;
#   Pleural Thickening barely moves whatever is present (AUC 0.542) so it stays
#   low — which is the correct behaviour, not a flaw.

MIN_SPREAD = 0.05


def salience(disease, probability, baselines):
    """How unusual this score is FOR THIS FINDING.

    z = (p - median) / spread, where spread is the finding's own measured
    (p90 - median). Roughly: 0 = a completely ordinary value, 1 = at the 90th
    percentile of ordinary films, >2 = genuinely unusual.

    SPREAD FLOOR — this matters.
      A near-constant finding has a tiny denominator, so trivial noise produces
      enormous salience. Measured on 400 healthy films, Hernia's spread is 0.017
      and Pleural Thickening's is 0.013: both output essentially the same number
      on every image. Without a floor, a 0.02 fluctuation in Hernia scored
      salience 1.2 and beat a genuinely elevated Cardiomegaly. Hernia was in
      fact the commonest false positive on healthy chests.
      A finding that never varies carries no information and must not be
      allowed to win the ranking through noise.

    Findings with no baseline fall back to raw probability so they still rank
    sensibly rather than vanishing.
    """
    b = baselines.get(disease)
    if not b:
        return float(probability)
    med = float(b.get("median", 0.0))
    # prefer the stored effective_spread (already floored by the generator);
    # fall back to computing it here so older baseline files still behave
    spread = b.get("effective_spread")
    if spread is None:
        spread = max(float(b.get("p90", 0.0)) - med, MIN_SPREAD)
    return (float(probability) - med) / max(float(spread), MIN_SPREAD)


def rank_by_salience(predictions, baselines, key="probability"):
    """Sort predictions most-salient first, annotating each one.

    Adds `salience` and `baseline_median`. The DISPLAYED probability is never
    modified — this changes ordering only. Showing a clinician a rescaled number
    would be worse than showing an honest one in a sensible order.
    """
    out = []
    for p in predictions:
        q = dict(p)
        b = baselines.get(p["disease"]) or {}
        q["baseline_median"] = b.get("median")
        q["salience"] = round(salience(p["disease"], p.get(key, 0.0), baselines), 3)
        out.append(q)
    return sorted(out, key=lambda x: -x["salience"])
