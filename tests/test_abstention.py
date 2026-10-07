# Tests for core/clinical_extras.py abstention logic.
#
# The ensemble's per-finding uncertainty is built in core/model_loader.predict as
#   sqrt(within^2 + between^2), where between = np.std(per-model probabilities).
# model_loader needs torch, so these tests compute `between` the same way from
# synthetic per-model outputs (within = 0, i.e. no MC-Dropout passes) and feed it
# to the real abstention functions.
import numpy as np

import clinical_extras as cx


def _finding(name, per_model_probs):
    probs = np.asarray(per_model_probs)
    return {"disease": name, "probability": float(probs.mean()),
            "uncertainty": float(np.std(probs))}


def test_abstains_when_models_disagree():
    # Four models split ~0.9 vs ~0.1 on every top finding -> the study is refused.
    preds = [_finding("Pneumonia",     [0.90, 0.10, 0.85, 0.15]),
             _finding("Consolidation", [0.80, 0.20, 0.75, 0.10]),
             _finding("Effusion",      [0.70, 0.05, 0.90, 0.20])]
    verdict = cx.panel_abstention(preds)
    assert verdict["abstain"] is True
    assert verdict["level"] == "study"


def test_does_not_abstain_when_models_agree():
    # Four models within a few points of each other -> the system answers.
    preds = [_finding("Cardiomegaly", [0.91, 0.93, 0.92, 0.90]),
             _finding("Effusion",     [0.20, 0.22, 0.18, 0.21]),
             _finding("Edema",        [0.10, 0.12, 0.11, 0.09])]
    verdict = cx.panel_abstention(preds)
    assert verdict["abstain"] is False
    assert not any(f["abstain"] for f in verdict["per_finding"])


def test_nan_uncertainty_is_not_treated_as_confident():
    # Edge case: a NaN uncertainty (e.g. a model emitting NaN) must not yield a confident answer.
    r = cx.should_abstain(0.50, float("nan"))
    assert r["abstain"] is True
