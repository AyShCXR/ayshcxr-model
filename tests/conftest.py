# Shared setup for the AyShCXR pytest suite.
# No GPU, no model weights, no datasets: every input is a small synthetic array.
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# core/ modules import each other by bare name (e.g. `import calibration`)
sys.path.insert(0, str(ROOT / "core"))


def to_image(a):
    """float array in [0,1] -> 8-bit greyscale PIL image."""
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


@pytest.fixture
def chest_array():
    """256x256 synthetic frontal 'chest': bright midline, two dark lung fields,
    fine texture noise. Fixed seed, so the result is identical on every run."""
    rng = np.random.default_rng(0)
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.full((h, w), 0.70)
    a += 0.25 * np.exp(-((xx - w / 2) ** 2) / (2 * (w * 0.07) ** 2))      # spine
    for cx in (w * 0.28, w * 0.72):                                        # lungs
        a -= 0.65 * np.exp(-((xx - cx) ** 2 / (2 * (w * 0.13) ** 2)
                             + (yy - h * 0.45) ** 2 / (2 * (h * 0.22) ** 2)))
    a += rng.normal(0, 0.03, (h, w))                                       # texture
    return np.clip(a, 0, 1)
