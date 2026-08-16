# symptom_fusion.py
# AyShCXR — Bayesian symptom fusion. Replaces the leaky FiLM approach.
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS EXISTS
#   The FiLM model (archive/superseded/train_symptom_fusion.py) scored 0.9085 val
#   AUC, and that number was meaningless: its symptom vectors were GENERATED FROM
#   THE DISEASE LABELS via SYMPTOM_PREVALENCE_BY_DISEASE, so the input contained
#   the answer. With uninformative symptoms the same model scored 0.6878 — worse
#   than the image alone. Retraining that design would only recreate the leak.
#
#   This module takes the opposite approach: NO TRAINING AT ALL. The image model's
#   probability is a prior; each reported symptom updates it by a likelihood ratio
#   derived from published prevalence data. Leakage is structurally impossible
#   because nothing is fitted to labels.
#
# THE MATH  (standard diagnostic Bayes, in log-odds space)
#     LR+ = P(S|D) / P(S|¬D)              symptom present
#     LR- = (1-P(S|D)) / (1-P(S|¬D))      symptom absent
#     logit(posterior) = logit(prior) + λ · Σ log(LR_i)
#
#   λ damps the naive-Bayes independence assumption. Fever and cough co-occur, so
#   multiplying their LRs at full strength double-counts one underlying signal.
#
# WHAT THIS DOES AND DOESN'T DO
#   Does    : change the probability shown to the clinician, with a visible audit
#             trail of which symptom moved it and by how much.
#   Doesn't : improve AUC. AUC scores the image model on images. Proving symptom
#             fusion improves accuracy needs a test set with real patient symptoms
#             paired to real radiographs — NIH and CheXpert have no such data.
#             Do NOT report an AUC gain from this module.
#
# Self-test:  python core/symptom_fusion.py

import math
from typing import Dict, Optional, List, Tuple

from medical_knowledge import SYMPTOM_PREVALENCE_BY_DISEASE

try:
    from disease_ontology import CANONICAL_FINDINGS
except ImportError:                       # usable standalone
    CANONICAL_FINDINGS = {}

# ── tuning constants ────────────────────────────────────────────────────────
LAMBDA        = 0.5     # independence damping. 1.0 = full naive Bayes (over-confident)
RISK_LAMBDA   = 0.35    # history (occupation/comorbidity) is damped harder than
                        # symptoms: it is indirect evidence, and its likelihood
                        # ratios are literature estimates rather than measured
                        # values. See core/risk_factors.py.
LR_CLAMP      = 10.0    # no single symptom may swing odds by more than this
POST_MIN      = 0.01    # never claim absolute certainty either way
POST_MAX      = 0.99
PREV_FLOOR    = 1e-4    # guards divide-by-zero on P(S|¬D)

# NIH ChestX-ray14 positive counts (112,120 images). Used to weight the estimate
# of P(symptom | NOT disease) — a rare disease should barely influence the
# background rate. Overridden at runtime if data/nih_full_labels.csv is present.
NIH_POSITIVE_COUNTS = {
    "Atelectasis": 11559, "Cardiomegaly": 2776, "Effusion": 13317,
    "Infiltration": 19894, "Mass": 5782, "Nodule": 6331, "Pneumonia": 1431,
    "Pneumothorax": 5302, "Consolidation": 4667, "Edema": 2303,
    "Emphysema": 2516, "Fibrosis": 1686, "Pleural Thickening": 3385,
    "Hernia": 227,
}


# ── odds helpers ────────────────────────────────────────────────────────────
def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── likelihood-ratio table ──────────────────────────────────────────────────
def _background_rate(symptom: str, exclude: str,
                     prevalence: Dict[str, float]) -> float:
    """P(symptom | NOT disease), estimated as the prevalence-weighted mean of
    P(symptom | other disease).

    ASSUMPTION, stated plainly: this averages over other DISEASES only; it does
    not model symptom rates in healthy people. Healthy patients report symptoms
    less often, so the true P(S|¬D) is lower and the true LRs are STRONGER than
    what this computes. The estimate is therefore conservative — it understates
    how much symptoms should move the probability. For a medical tool, erring
    toward smaller adjustments is the right direction to be wrong in.
    """
    num = den = 0.0
    for d, table in SYMPTOM_PREVALENCE_BY_DISEASE.items():
        if d == exclude or symptom not in table:
            continue
        w = prevalence.get(d, 0.0)
        num += w * float(table[symptom])
        den += w
    return max(num / den, PREV_FLOOR) if den else PREV_FLOOR


def build_lr_table(prevalence: Optional[Dict[str, float]] = None
                   ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """{disease: {symptom: (LR_present, LR_absent)}}"""
    if prevalence is None:
        total = sum(NIH_POSITIVE_COUNTS.values())
        prevalence = {d: c / total for d, c in NIH_POSITIVE_COUNTS.items()}

    table: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for disease, symptoms in SYMPTOM_PREVALENCE_BY_DISEASE.items():
        entry = {}
        for symptom, p_s_given_d in symptoms.items():
            if not isinstance(p_s_given_d, (int, float)):
                continue                                    # skip prose fields
            p_d = min(max(float(p_s_given_d), PREV_FLOOR), 1 - PREV_FLOOR)
            p_nd = min(max(_background_rate(symptom, disease, prevalence),
                           PREV_FLOOR), 1 - PREV_FLOOR)
            lr_pos = p_d / p_nd
            lr_neg = (1 - p_d) / (1 - p_nd)
            entry[symptom] = (
                min(max(lr_pos, 1 / LR_CLAMP), LR_CLAMP),
                min(max(lr_neg, 1 / LR_CLAMP), LR_CLAMP),
            )
        table[disease] = entry
    return table


LR_TABLE = build_lr_table()


# ── the update ──────────────────────────────────────────────────────────────
def fuse_one(disease: str, image_prob: float, symptoms: Dict[str, Optional[bool]],
             lam: float = LAMBDA) -> dict:
    """Update one disease's probability given reported symptoms.

    symptoms: {key: True (present) | False (absent) | None (not asked)}
              None and unknown keys contribute nothing — absence of evidence is
              not evidence of absence.
    """
    kind = CANONICAL_FINDINGS.get(disease, {}).get("kind", "disease")
    if kind in ("device", "normal"):
        # "Support Devices" / "No Finding" are not pathologies; symptoms are
        # meaningless for them and must not shift their probability.
        return {"disease": disease, "prior": image_prob, "posterior": image_prob,
                "delta": 0.0, "evidence": [], "skipped": f"kind={kind}"}

    lrs = LR_TABLE.get(disease)
    if not lrs:
        # e.g. the CheXpert-only findings, which have no prevalence data yet
        return {"disease": disease, "prior": image_prob, "posterior": image_prob,
                "delta": 0.0, "evidence": [], "skipped": "no prevalence data"}

    log_odds = _logit(image_prob)
    evidence: List[dict] = []
    total_log_lr = 0.0

    for symptom, answer in symptoms.items():
        if answer is None or symptom not in lrs:
            continue
        lr = lrs[symptom][0 if answer else 1]
        total_log_lr += math.log(lr)
        evidence.append({
            "symptom": symptom,
            "present": bool(answer),
            "lr": round(lr, 3),
            "direction": "supports" if lr > 1 else ("against" if lr < 1 else "neutral"),
        })

    posterior = _sigmoid(log_odds + lam * total_log_lr)
    posterior = min(max(posterior, POST_MIN), POST_MAX)

    evidence.sort(key=lambda e: abs(math.log(e["lr"])), reverse=True)
    return {
        "disease": disease,
        "prior": round(image_prob, 4),
        "posterior": round(posterior, 4),
        "delta": round(posterior - image_prob, 4),
        "evidence": evidence,
        "skipped": None,
    }


def fuse(image_probs: Dict[str, float], symptoms: Dict[str, Optional[bool]],
         lam: float = LAMBDA,
         risk_lrs: Optional[Dict[str, float]] = None) -> Dict[str, dict]:
    """Apply the update across every predicted finding.

    risk_lrs: optional {disease: likelihood_ratio} from occupation and
    comorbidity history (see core/risk_factors.py). Applied in the SAME
    log-odds update as symptoms rather than as a separate additive boost, so
    all evidence combines correctly and nothing is counted twice.
    """
    out = {}
    for d, p in image_probs.items():
        r = fuse_one(d, p, symptoms, lam)
        lr = (risk_lrs or {}).get(d)
        if lr and lr != 1.0 and not r["skipped"]:
            # history is weaker evidence than a directly observed symptom, and
            # these LRs are literature estimates rather than measured values —
            # so damp them harder than LAMBDA does for symptoms.
            adj = _sigmoid(_logit(r["posterior"]) + RISK_LAMBDA * math.log(lr))
            r["risk_lr"] = round(lr, 2)
            r["risk_delta"] = round(adj - r["posterior"], 4)
            r["posterior"] = round(min(max(adj, POST_MIN), POST_MAX), 4)
            r["delta"] = round(r["posterior"] - r["prior"], 4)
        out[d] = r
    return out


def explain(result: dict, top_n: int = 5) -> str:
    """One-line-per-symptom audit trail for the UI / clinical report."""
    if result["skipped"]:
        return f"{result['disease']}: unchanged ({result['skipped']})"
    if not result["evidence"]:
        return (f"{result['disease']}: {result['prior']:.0%} "
                f"(no symptoms reported — image only)")
    parts = [f"{e['symptom']}{'' if e['present'] else ' (absent)'} "
             f"x{e['lr']:.2f}" for e in result["evidence"][:top_n]]
    return (f"{result['disease']}: {result['prior']:.0%} -> "
            f"{result['posterior']:.0%}  [" + ", ".join(parts) + "]")


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("  AyShCXR — Bayesian symptom fusion self-test")
    print("=" * 72)

    print(f"\ndiseases with prevalence data : {len(LR_TABLE)}")
    print(f"symptom keys per disease      : {len(next(iter(LR_TABLE.values())))}")
    print(f"lambda (independence damping) : {LAMBDA}")

    # which symptoms actually discriminate for Pneumonia?
    pn = sorted(LR_TABLE["Pneumonia"].items(),
                key=lambda kv: abs(math.log(kv[1][0])), reverse=True)
    print("\nmost informative symptoms for Pneumonia (LR when present):")
    for s, (lp, ln) in pn[:6]:
        print(f"   {s:24s} LR+ {lp:6.2f}   LR- {ln:5.2f}")
    print("\nleast informative (near 1.0 = tells you almost nothing):")
    for s, (lp, ln) in pn[-4:]:
        print(f"   {s:24s} LR+ {lp:6.2f}   LR- {ln:5.2f}")

    # realistic case: image is uncertain, patient looks like pneumonia
    print("\n" + "-" * 72)
    print("CASE 1 — image uncertain (40%), classic pneumonia presentation")
    print("-" * 72)
    case = {"fever_high": True, "cough_productive": True,
            "breathless_present": True, "pleuritic_pain": True,
            "chest_pain": False, "weight_loss": False, "night_sweats": None}
    r = fuse_one("Pneumonia", 0.40, case)
    print(explain(r))
    for e in r["evidence"]:
        arrow = "^" if e["lr"] > 1 else "v"
        print(f"   {arrow} {e['symptom']:22s} "
              f"{'present' if e['present'] else 'absent ':8s} x{e['lr']:.2f}")
    print(f"\n   {r['prior']:.1%} -> {r['posterior']:.1%}  "
          f"(delta {r['delta']:+.1%})")

    # same image probability, symptoms pointing away
    print("\n" + "-" * 72)
    print("CASE 2 — same 40% image, but symptoms argue AGAINST pneumonia")
    print("-" * 72)
    case2 = {"fever_high": False, "cough_productive": False,
             "breathless_present": False, "pleuritic_pain": False}
    r2 = fuse_one("Pneumonia", 0.40, case2)
    print(f"   {r2['prior']:.1%} -> {r2['posterior']:.1%}  "
          f"(delta {r2['delta']:+.1%})")
    print("   NOTE: the old 90/10 blend could not push a probability DOWN "
          "on negative findings. This can.")

    # no symptoms at all -> must be a no-op
    print("\n" + "-" * 72)
    print("CASE 3 — no symptoms reported (image only)")
    print("-" * 72)
    r3 = fuse_one("Pneumonia", 0.40, {"fever_high": None})
    print(f"   {r3['prior']:.1%} -> {r3['posterior']:.1%}  "
          f"(delta {r3['delta']:+.1%})")
    assert r3["posterior"] == r3["prior"], "no evidence must mean no change"

    # non-pathology findings must never move
    if CANONICAL_FINDINGS:
        print("\n" + "-" * 72)
        print("CASE 4 — non-pathology findings are skipped")
        print("-" * 72)
        for nd in ("Support Devices", "No Finding"):
            rr = fuse_one(nd, 0.50, case)
            print(f"   {nd:16s} {rr['prior']:.0%} -> {rr['posterior']:.0%}  "
                  f"({rr['skipped']})")
            assert rr["delta"] == 0.0

    # whole-panel example
    print("\n" + "-" * 72)
    print("CASE 5 — full panel, same patient")
    print("-" * 72)
    panel = {"Pneumonia": 0.40, "Consolidation": 0.35, "Effusion": 0.20,
             "Cardiomegaly": 0.15, "Hernia": 0.02}
    for d, res in fuse(panel, case).items():
        bar = "#" * int(res["posterior"] * 40)
        print(f"   {d:16s} {res['prior']:.0%} -> {res['posterior']:5.1%} "
              f"{res['delta']:+6.1%}  {bar}")

    print("\n" + "=" * 72)
    print("  ALL CHECKS PASSED")
    print("  Reminder: this changes displayed probabilities and explains them.")
    print("  It does NOT improve AUC and must never be reported as doing so.")
    print("=" * 72)
