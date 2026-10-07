# Tests for core/image_quality.py — film technique checks.
import numpy as np

import image_quality as iq
from conftest import to_image


def _check(report, name):
    return next(c for c in report["checks"] if c["check"] == name)


def _severity(report, name):
    return _check(report, name)["severity"]


def test_normal_image_passes(chest_array):
    # A well-exposed, sharp, symmetric synthetic chest gets an overall "pass".
    r = iq.assess(to_image(chest_array))
    assert r["available"] is True
    assert r["overall"] == "pass"
    assert r["warnings"] == []


def test_blurry_image_flagged_severe(chest_array):
    # Repeated box-blurring removes edge energy, so the sharpness check fires.
    a = chest_array.copy()
    for _ in range(6):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    r = iq.assess(to_image(a))
    assert _severity(r, "sharpness") == "severe"
    assert r["overall"] == "severe"


def test_too_dark_and_too_bright_images_flagged(chest_array):
    # Over-penetrated (dark) and under-penetrated (bright) films both fail exposure.
    dark = iq.assess(to_image(chest_array * 0.4))
    bright = iq.assess(to_image(chest_array * 0.4 + 0.6))
    assert _severity(dark, "exposure") == "severe"
    assert _severity(bright, "exposure") == "severe"
    assert "over-penetrated" in _check(dark, "exposure")["message"]
    assert "under-penetrated" in _check(bright, "exposure")["message"]


def test_corrupted_input_reported_unavailable_without_crashing():
    # Non-image input must not raise; assess() reports it as unavailable instead.
    r = iq.assess(b"\x89PNG\r\n\x1a\n truncated garbage")
    assert r["available"] is False
    assert r["overall"] == "unknown"
    assert r["checks"] == []
