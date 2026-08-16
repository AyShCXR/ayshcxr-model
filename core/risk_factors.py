# risk_factors.py
# AyShCXR — occupation and comorbidity risk factors as likelihood ratios.
# by Subhrakant Sethi & Ayush Singh — 2026-08-10
#
# WHY THIS REPLACES apply_history_boost()
#   The old version added a flat bonus to the probability: coal miner ->
#   Fibrosis +0.08, hypertension -> Cardiomegaly +0.07, capped at +0.10 total.
#   Four things were wrong with that:
#
#   1. WRONG SCALE. Those constants were chosen when scores sat around 0.4-0.5.
#      On the current merged+calibrated pipeline the medians are Cardiomegaly
#      0.052 and Edema 0.039 — so typing "hypertension" into a text box MORE
#      THAN DOUBLED the Cardiomegaly probability before the X-ray was even
#      considered. The patient history was overwhelming the image.
#   2. ADDITIVE, not probabilistic. Everything else in this system works in
#      log-odds precisely so evidence combines correctly at any starting point.
#   3. DOUBLE COUNTING. It boosted Effusion for `symptoms["swelling"]`, while
#      symptom_fusion ALSO applied a likelihood ratio to that same swelling.
#      Identical evidence counted twice. Same for tb_contact, night_sweats,
#      haemoptysis, recent_surgery and fever.
#   4. NAIVE TEXT MATCHING. `"dm" in conditions` matched "aDMitted" and
#      "oeDeMa"; `"sugar"` matched "no sugar"; and nothing detected negation, so
#      "no history of diabetes" fired the diabetes boost.
#
#   Risk factors are now evidence like any other: parsed with word boundaries,
#   negation-aware, and applied as likelihood ratios through the same Bayesian
#   update. Symptoms are deliberately NOT handled here — symptom_fusion owns
#   those, so nothing is counted twice.
#
# ── PROVENANCE OF THE NUMBERS — read this ──────────────────────────────────
#   The likelihood ratios below are ORDER-OF-MAGNITUDE CLINICAL ESTIMATES drawn
#   from the epidemiological literature (relative risks for occupational lung
#   disease, comorbidity associations). They are NOT fitted to your data and
#   NOT measured on any validation set, because no dataset here pairs
#   radiographs with occupational histories.
#
#   They are deliberately CONSERVATIVE — a real asbestos-fibrosis association is
#   stronger than the 4.0 used here. Under-weighting keeps the image dominant,
#   which is the correct failure direction for an imaging tool.
#
#   Do not present these as validated. They encode "this exposure makes this
#   disease meaningfully more likely", nothing finer.
#
# Self-test:  python core/risk_factors.py

import re
from typing import Dict, List

# ── risk factor definitions ────────────────────────────────────────────────
# term patterns are matched on WORD BOUNDARIES, so "dm" no longer matches
# "admitted" and "sugar" no longer matches inside another word.
RISK_FACTORS = {
    "asbestos": {
        "field": "occupation",
        "terms": [r"asbestos", r"shipyard", r"insulat\w*", r"lagging"],
        "lr": {"Fibrosis": 4.0, "Pleural Other": 5.0,
               "Pleural Thickening": 4.5, "Lung Lesion": 2.0},
        "note": "Asbestos exposure — pleural plaques, thickening and fibrosis",
    },
    "coal_mining": {
        "field": "occupation",
        "terms": [r"coal", r"\bminer?\b", r"\bmining\b", r"colliery"],
        "lr": {"Fibrosis": 4.0, "Emphysema": 2.0, "Lung Opacity": 1.8},
        "note": "Coal dust exposure — pneumoconiosis risk",
    },
    "silica": {
        "field": "occupation",
        "terms": [r"quarry\w*", r"sandblast\w*", r"stone\s*cut\w*", r"silica",
                  r"\bgranite\b", r"foundry"],
        "lr": {"Fibrosis": 4.0, "Lung Opacity": 1.8},
        "note": "Silica exposure — silicosis risk, also raises TB risk",
    },
    "welding_metal": {
        "field": "occupation",
        "terms": [r"welder", r"welding", r"\bmetal\s*work\w*"],
        "lr": {"Fibrosis": 2.0, "Lung Opacity": 1.5},
        "note": "Metal fume exposure",
    },
    "biomass_textile": {
        "field": "occupation",
        "terms": [r"textile", r"cotton", r"\bmill\b", r"biomass",
                  r"chulha", r"cook\w*\s*(fire|stove)"],
        "lr": {"Emphysema": 2.0, "Fibrosis": 1.6},
        "note": "Organic dust or biomass smoke — a major rural Indian exposure",
    },

    "heart_failure": {
        "field": "conditions",
        "terms": [r"heart\s*failure", r"\bccf\b", r"\bchf\b", r"cardiomyopath\w*",
                  r"\bhfref\b", r"\bhfpef\b"],
        "lr": {"Edema": 5.0, "Effusion": 4.0, "Cardiomegaly": 5.0},
        "note": "Known heart failure",
    },
    "hypertension": {
        "field": "conditions",
        "terms": [r"hypertens\w*", r"\bhtn\b", r"high\s*bp",
                  r"high\s*blood\s*pressure"],
        "lr": {"Cardiomegaly": 2.5, "Enlarged Cardiomediastinum": 2.0},
        "note": "Hypertension — left ventricular hypertrophy over time",
    },
    "renal": {
        "field": "conditions",
        "terms": [r"renal\s*failure", r"\bckd\b", r"nephrotic", r"dialysis",
                  r"kidney\s*(failure|disease)"],
        "lr": {"Edema": 4.0, "Effusion": 3.0},
        "note": "Renal disease — fluid overload",
    },
    "immunosuppressed": {
        "field": "conditions",
        "terms": [r"\bhiv\b", r"\baids\b", r"immunocompromis\w*",
                  r"immunosuppress\w*", r"chemotherapy", r"transplant"],
        "lr": {"Pneumonia": 3.0, "Consolidation": 2.5, "Lung Opacity": 2.0,
               "Infiltration": 2.0},
        "note": "Immunosuppression — higher risk of infection, including TB",
    },
    "diabetes": {
        "field": "conditions",
        "terms": [r"diabet\w*", r"\bdm\b", r"\bt2dm\b", r"\biddm\b",
                  r"\bniddm\b"],
        "lr": {"Pneumonia": 1.8, "Consolidation": 1.5},
        "note": "Diabetes — increased infection risk, including TB",
    },
    "copd": {
        "field": "conditions",
        "terms": [r"\bcopd\b", r"emphysema", r"chronic\s*bronchitis"],
        "lr": {"Emphysema": 4.0, "Pneumothorax": 2.5},
        "note": "Known COPD",
    },
    "prior_tb": {
        "field": "conditions",
        "terms": [r"\btb\b", r"tuberculosis", r"\bptb\b", r"\bdots\b"],
        "lr": {"Fibrosis": 3.0, "Lung Lesion": 2.5, "Pleural Thickening": 2.5,
               "Infiltration": 2.0},
        "note": "Previous tuberculosis — scarring and cavitation",
    },
    "malignancy": {
        "field": "conditions",
        "terms": [r"cancer", r"carcinoma", r"malignan\w*", r"metasta\w*",
                  r"\bca\s+(lung|breast)"],
        "lr": {"Lung Lesion": 3.5, "Mass": 3.5, "Nodule": 2.5, "Effusion": 2.0},
        "note": "Known malignancy — metastatic disease possible",
    },

    "smoking_current": {
        "field": "smoking",
        "terms": [r"^(yes|current|heavy)$"],
        "lr": {"Emphysema": 4.0, "Lung Lesion": 3.0, "Mass": 3.0,
               "Nodule": 2.0, "Fibrosis": 1.5},
        "note": "Current smoker",
    },
    "smoking_past": {
        "field": "smoking",
        "terms": [r"^(past|ex|former)$"],
        "lr": {"Emphysema": 2.5, "Lung Lesion": 2.0, "Mass": 2.0,
               "Nodule": 1.6},
        "note": "Ex-smoker — risk falls but does not return to baseline",
    },
}

# words that negate a following term within a short window
NEGATIONS = (r"no", r"not", r"non", r"nil", r"denies", r"denied", r"without",
             r"negative\s+for", r"absent", r"never", r"ruled?\s+out", r"free\s+of")
NEG_WINDOW = 25          # characters before the match to inspect


def _is_negated(text: str, start: int) -> bool:
    """Is the match at `start` preceded by a negation?

    Deliberately simple: a short look-back window, not a parser. It catches the
    realistic clinical shorthand ("no dm", "denies tb contact", "non-smoker")
    that would otherwise fire a risk factor for a condition the patient has
    explicitly been recorded as NOT having.
    """
    window = text[max(0, start - NEG_WINDOW):start]
    return bool(re.search(r"(?:^|\W)(?:" + "|".join(NEGATIONS) + r")\W+$", window))


def parse(occupation: str = "", conditions: str = "",
          smoking: str = "") -> List[dict]:
    """Free-text history -> list of detected risk factors.

    Returns [{key, note, lr, matched, field}] — never raises.
    """
    fields = {
        "occupation": (occupation or "").lower().strip(),
        "conditions": (conditions or "").lower().strip(),
        "smoking":    (smoking or "").lower().strip(),
    }
    found = []
    for key, spec in RISK_FACTORS.items():
        text = fields.get(spec["field"], "")
        if not text:
            continue
        for pat in spec["terms"]:
            m = re.search(pat, text)
            if not m:
                continue
            if spec["field"] != "smoking" and _is_negated(text, m.start()):
                continue                      # "no diabetes" must not fire
            found.append({"key": key, "note": spec["note"], "lr": spec["lr"],
                          "matched": m.group(0), "field": spec["field"]})
            break
    return found


def likelihood_ratios(risk_list: List[dict]) -> Dict[str, float]:
    """Combine detected risk factors into one LR per finding.

    Two exposures affecting the same disease multiply, but the product is capped:
    a patient who is a smoking coal miner is genuinely at higher risk, yet three
    stacked risk factors must not swamp the radiograph. The image is the primary
    evidence; history adjusts it.
    """
    CAP = 6.0
    out: Dict[str, float] = {}
    for r in risk_list:
        for disease, lr in r["lr"].items():
            out[disease] = out.get(disease, 1.0) * float(lr)
    return {d: min(v, CAP) for d, v in out.items()}


if __name__ == "__main__":
    print("=" * 72)
    print("  risk_factors self-test")
    print("=" * 72)

    cases = [
        ("coal miner, 30 years", "", "yes"),
        ("welder", "hypertension, diabetes", "no"),
        ("", "no diabetes, no tb", ""),
        ("", "denies hypertension", ""),
        ("office worker", "known CCF", "past"),
        ("", "admitted last year", ""),          # must NOT match "dm"
        ("", "oedema of ankles", ""),            # must NOT match "dm"
        ("", "patient has no sugar problem", ""),  # must NOT match diabetes
        ("asbestos insulation fitter", "old PTB", "yes"),
    ]
    for occ, cond, smk in cases:
        rl = parse(occ, cond, smk)
        keys = [r["key"] for r in rl]
        lrs = likelihood_ratios(rl)
        top = sorted(lrs.items(), key=lambda kv: -kv[1])[:3]
        print(f"\n  occ={occ!r:28s} cond={cond!r:26s} smoke={smk!r}")
        print(f"    detected : {keys or 'none'}")
        if top:
            print(f"    top LRs  : {', '.join(f'{d} x{v:.1f}' for d, v in top)}")

    print("\n  negation checks:")
    assert not parse("", "no diabetes", ""), "negation failed: 'no diabetes'"
    assert not parse("", "denies hypertension", ""), "negation failed: 'denies'"
    assert not parse("", "admitted last year", ""), "'dm' matched inside 'admitted'"
    assert not parse("", "oedema of ankles", ""), "'dm' matched inside 'oedema'"
    assert not parse("", "patient has no sugar problem", ""), "'no sugar' fired"
    assert parse("", "diabetes mellitus", ""), "positive case missed"
    assert parse("coal miner", "", ""), "occupation missed"
    print("    all passed")

    print("\n  cap check (smoking coal miner with COPD):")
    rl = parse("coal miner", "copd", "yes")
    lrs = likelihood_ratios(rl)
    print(f"    Emphysema LR = {lrs.get('Emphysema'):.1f}  (capped at 6.0)")
    assert lrs.get("Emphysema", 0) <= 6.0

    print("\n" + "=" * 72)
    print("  ALL CHECKS PASSED")
    print("  LRs are literature-informed estimates, NOT fitted or validated.")
    print("=" * 72)
