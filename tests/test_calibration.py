# Tests for core/calibration.py — per-finding temperature scaling.
import numpy as np

import calibration as calib
from conftest import ROOT

CALIB_FILE = str(ROOT / "results" / "calibration.json")   # committed, 1 KB


def test_calibrated_probabilities_stay_between_0_and_1():
    # Even extreme raw scores (exactly 0 or 1) map into the valid [0, 1] range.
    raw = [0.0, 1e-9, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0]
    for finding in ("Cardiomegaly", "Lung Opacity", "Pneumonia"):
        for p in raw:
            out = calib.calibrate(finding, p, path=CALIB_FILE)
            assert 0.0 <= out <= 1.0, (finding, p, out)


def test_temperature_scaling_keeps_highest_ranked_case_and_order():
    # For one finding, scaling must not change which case scores highest (AUC preserved).
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.01, 0.99, size=50)
    cal = np.array([calib.calibrate("Cardiomegaly", p, path=CALIB_FILE) for p in raw])
    assert np.argmax(cal) == np.argmax(raw)
    assert np.array_equal(np.argsort(cal), np.argsort(raw))


def test_effusion_is_calibrated_under_its_canonical_name():
    # The app asks for "Effusion"; the file stores "Pleural Effusion" — both must be calibrated.
    assert calib.calibrate("Effusion", 0.6, path=CALIB_FILE) != 0.6
    assert calib.calibrate("Effusion", 0.6, path=CALIB_FILE) == \
        calib.calibrate("Pleural Effusion", 0.6, path=CALIB_FILE)
