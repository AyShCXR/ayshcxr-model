# Tests for core/finding_reliability.py — the AUC >= 0.70 reporting bar.
import finding_reliability as rel


def test_findings_below_auc_bar_are_withheld_and_others_reported():
    # Below 0.70 (or never measured) -> withheld with a message; at/above -> reported.
    assert rel.MIN_DIAGNOSTIC_AUC == 0.70
    for f in ("Pleural Thickening", "Fibrosis"):       # 0.497 / 0.643 (lower AUC governs)
        assert rel.status(f) == "unreliable"
        assert not rel.can_diagnose(f)
        assert rel.withhold_message(f)["finding"] == f
    assert rel.status("Fracture") == "unvalidated"     # no test data at all
    assert not rel.can_diagnose("Fracture")
    for f in ("Cardiomegaly", "Effusion", "Lung Opacity"):   # Lung Opacity = exactly 0.700
        assert rel.can_diagnose(f)
        assert rel.withhold_message(f) is None
