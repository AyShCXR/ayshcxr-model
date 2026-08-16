# finding_reliability.py
# AyShCXR — what the system is and is not entitled to diagnose. 2026-08-10
# by Subhrakant Sethi & Ayush Singh
#
# THE PRINCIPLE
#   All 21 findings are still analysed. But a finding may only be REPORTED AS A
#   DIAGNOSIS if it has been measured against labelled data and reached an AUC
#   of at least 0.70. Anything below that, or never measured, is withheld — and
#   the system says so plainly instead of guessing.
#
#   This is not the system doing less. It is the system refusing to state
#   things it cannot support. A tool that says "I cannot assess pleural
#   thickening reliably" is more useful in a clinic than one that reports it at
#   44% and is right by coincidence.
#
# THE EVIDENCE
#   NIH ChestX-ray14 — 1,370 stratified images, all 14 NIH findings, plus 250
#   films with no reported finding. Ground truth is NLP-mined from radiology
#   reports (~90% accurate) — the same KIND of label the models trained on, so
#   these numbers are the optimistic end.
#
#   VinDr-CXR — 400-1,300 images, radiologist-read (>=2 of 3 agreement). Never
#   seen in training, different country and hospitals. Harder and more honest.
#
#   Where the two disagree the LOWER value is used. Fibrosis measures 0.804 on
#   NIH and 0.643 on VinDr; it is recorded as 0.643, because a finding that
#   only works when graded by the same kind of software that trained it has not
#   been shown to work.
#
# Self-test:  python core/finding_reliability.py

MIN_DIAGNOSTIC_AUC = 0.70

# auc_nih / auc_vindr are as measured; `auc` is the value used for the decision.
# top1 = how often this finding was named FIRST on a film where it was the only
# finding present. n = positives in the test sample.
EVIDENCE = {
    # ── NIH-measurable findings ────────────────────────────────────────────
    "Emphysema":        {"auc_nih": 0.911, "auc_vindr": None, "n": 111, "top1": 0.77},
    "Hernia":           {"auc_nih": 0.911, "auc_vindr": None, "n":  80, "top1": 0.84},
    "Cardiomegaly":     {"auc_nih": 0.914, "auc_vindr": 0.913, "n": 113, "top1": 0.53},
    "Pneumothorax":     {"auc_nih": 0.896, "auc_vindr": None, "n": 162, "top1": 0.52},
    "Edema":            {"auc_nih": 0.883, "auc_vindr": None, "n": 119, "top1": 0.82},
    "Effusion":         {"auc_nih": 0.829, "auc_vindr": 0.935, "n": 295, "top1": 0.33},
    "Mass":             {"auc_nih": 0.812, "auc_vindr": None, "n": 166, "top1": 0.36},
    "Atelectasis":      {"auc_nih": 0.791, "auc_vindr": None, "n": 249, "top1": 0.22},
    "Nodule":           {"auc_nih": 0.776, "auc_vindr": None, "n": 159, "top1": 0.62},
    "Pneumonia":        {"auc_nih": 0.745, "auc_vindr": None, "n": 105, "top1": 0.25},
    "Infiltration":     {"auc_nih": 0.732, "auc_vindr": 0.712, "n": 321, "top1": 0.02},
    "Consolidation":    {"auc_nih": 0.725, "auc_vindr": 0.908, "n": 156, "top1": 0.10},

    # Fibrosis: 0.804 on NIH but 0.643 against radiologists. The lower value
    # governs — see the note above.
    "Fibrosis":         {"auc_nih": 0.804, "auc_vindr": 0.643, "n": 100, "top1": 0.67},

    # Measured and genuinely broken. Sensitivity 99%, specificity 1%: it says
    # "yes" to almost every image. This is a constant, not a detector, and it is
    # what flagged every healthy chest in testing.
    "Pleural Thickening": {"auc_nih": 0.497, "auc_vindr": 0.542, "n": 146, "top1": 0.03},

    # ── CheXpert-only: VinDr provides partial evidence ─────────────────────
    "Lung Lesion":      {"auc_nih": None, "auc_vindr": 0.807, "n": 83, "top1": 0.22},
    "Lung Opacity":     {"auc_nih": None, "auc_vindr": 0.700, "n": 105, "top1": 0.04},

    # ── No ground truth in any dataset available to us ─────────────────────
    "Enlarged Cardiomediastinum": {"auc_nih": None, "auc_vindr": None, "n": 0, "top1": None},
    "Pleural Other":              {"auc_nih": None, "auc_vindr": None, "n": 0, "top1": None},
    "Fracture":                   {"auc_nih": None, "auc_vindr": None, "n": 0, "top1": None},

    # not diseases — handled separately by medical_knowledge_ext.is_pathology()
    "Support Devices":  {"auc_nih": None, "auc_vindr": None, "n": 0, "top1": None},
    "No Finding":       {"auc_nih": None, "auc_vindr": None, "n": 0, "top1": None},
}

NOT_A_DISEASE = ("Support Devices", "No Finding")


def _governing_auc(e):
    """The lower of the two measurements. A finding that only performs when
    graded by the same kind of labels it trained on has not been shown to work."""
    vals = [v for v in (e.get("auc_nih"), e.get("auc_vindr")) if v is not None]
    return min(vals) if vals else None


def status(finding):
    """One of: 'diagnostic', 'unreliable', 'unvalidated', 'not_a_disease'."""
    if finding in NOT_A_DISEASE:
        return "not_a_disease"
    e = EVIDENCE.get(finding)
    if not e:
        return "unvalidated"
    auc = _governing_auc(e)
    if auc is None:
        return "unvalidated"
    return "diagnostic" if auc >= MIN_DIAGNOSTIC_AUC else "unreliable"


def can_diagnose(finding):
    """May this finding be reported as a diagnosis?"""
    return status(finding) == "diagnostic"


def evidence_for(finding):
    e = EVIDENCE.get(finding, {})
    return {
        "finding": finding,
        "status": status(finding),
        "auc": _governing_auc(e),
        "auc_nih": e.get("auc_nih"),
        "auc_vindr": e.get("auc_vindr"),
        "n_tested": e.get("n", 0),
        "named_first": e.get("top1"),
    }


def withhold_message(finding, lang="en"):
    """What to tell the user when the image points at a finding we cannot back.

    Deliberately specific: it names the finding, says why, and says what to do.
    A vague "uncertain" teaches the operator to ignore the message.
    """
    st = status(finding)
    if st == "diagnostic":
        return None
    if st == "unreliable":
        e = evidence_for(finding)
        en = (f"This X-ray shows a pattern that may indicate {finding}, but "
              f"AyShCXR cannot assess {finding} reliably. In our validation it "
              f"performed no better than chance (AUC {e['auc']:.2f} on "
              f"{e['n_tested']} tested cases), so we do not report it as a "
              f"diagnosis. Please have a doctor review this film. We are "
              f"working to improve this finding with radiologist-labelled data.")
        hi = (f"इस एक्स-रे में {finding} जैसा पैटर्न दिख सकता है, लेकिन AyShCXR "
              f"इसका विश्वसनीय आकलन नहीं कर सकता। कृपया डॉक्टर को दिखाएं।")
    else:
        en = (f"This X-ray shows a pattern that may indicate {finding}. "
              f"AyShCXR has not yet been validated for {finding} — we have no "
              f"labelled test data for it, so we cannot state how accurate we "
              f"would be. We do not report it as a diagnosis. Please have a "
              f"doctor review this film. Validation for this finding is "
              f"planned.")
        hi = (f"इस एक्स-रे में {finding} जैसा पैटर्न दिख सकता है, लेकिन इस बीमारी के "
              f"लिए AyShCXR की जांच अभी नहीं हुई है। कृपया डॉक्टर को दिखाएं।")
    return {"en": en, "hi": hi, "finding": finding, "status": st}


def summary():
    """Grouped lists, for the UI and for the paper."""
    out = {"diagnostic": [], "unreliable": [], "unvalidated": [], "not_a_disease": []}
    for f in EVIDENCE:
        out[status(f)].append(f)
    return out


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 76)
    print("  AyShCXR — what may and may not be diagnosed")
    print("=" * 76)
    print(f"\n  Threshold for reporting a diagnosis: AUC >= {MIN_DIAGNOSTIC_AUC}")
    print("  Where NIH and VinDr disagree, the LOWER value governs.\n")

    print(f"{'finding':<28}{'NIH':>7}{'VinDr':>8}{'used':>7}{'n':>6}{'1st':>6}  status")
    print("-" * 76)
    for f, e in sorted(EVIDENCE.items(),
                       key=lambda kv: -(_governing_auc(kv[1]) or 0)):
        if f in NOT_A_DISEASE:
            continue
        g = _governing_auc(e)
        nih = f"{e['auc_nih']:.3f}" if e["auc_nih"] else "  -  "
        vin = f"{e['auc_vindr']:.3f}" if e["auc_vindr"] else "  -  "
        use = f"{g:.3f}" if g else "  -  "
        t1 = f"{e['top1']:.0%}" if e["top1"] is not None else "  - "
        print(f"{f:<28}{nih:>7}{vin:>8}{use:>7}{e['n']:>6}{t1:>6}  {status(f)}")
    print("-" * 76)

    s = summary()
    print(f"\n  CAN DIAGNOSE ({len(s['diagnostic'])}):")
    for f in s["diagnostic"]:
        print(f"     {f}")
    print(f"\n  WITHHELD — measured, not good enough ({len(s['unreliable'])}):")
    for f in s["unreliable"]:
        print(f"     {f}  (AUC {evidence_for(f)['auc']:.3f})")
    print(f"\n  WITHHELD — never tested ({len(s['unvalidated'])}):")
    for f in s["unvalidated"]:
        print(f"     {f}")

    print("\n  example message when a withheld finding is the strongest signal:")
    for f in (s["unreliable"] + s["unvalidated"])[:2]:
        m = withhold_message(f)
        print(f"\n   [{f}]")
        for line in [m["en"][i:i+70] for i in range(0, len(m["en"]), 70)]:
            print(f"     {line}")

    assert can_diagnose("Emphysema")
    assert not can_diagnose("Pleural Thickening")
    assert not can_diagnose("Fracture")
    assert status("No Finding") == "not_a_disease"
    print("\n" + "=" * 76)
    print("  ALL CHECKS PASSED")
    print("=" * 76)
