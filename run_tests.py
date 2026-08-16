#!/usr/bin/env python
# run_tests.py — AyShCXR system test suite. 2026-08-10
#
# Every core module carried a __main__ self-test but nothing ran them together,
# so a regression in one module was only found by chance. This runs all of them
# plus cross-module integration checks and an end-to-end HTTP request.
#
#   python run_tests.py           full suite (loads models, needs torch)
#   python run_tests.py --fast    skip anything that loads model weights
#
# Exit code 0 = all passed, 1 = at least one failure.

import argparse
import io
import os
import subprocess
import sys
import contextlib
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "core"))

PASS, FAIL, SKIP = [], [], []


def check(name, fn, needs_models=False, fast=False):
    if needs_models and fast:
        SKIP.append(name)
        print(f"  SKIP  {name}")
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {str(e)[:200]}")


# ── module self-tests ───────────────────────────────────────────────────────
def module_selftest(mod):
    def run():
        r = subprocess.run([sys.executable, f"core/{mod}.py"],
                           capture_output=True, text=True, timeout=600,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            raise AssertionError(f"exit {r.returncode}: {r.stderr[-300:]}")
    return run


# ── ontology ────────────────────────────────────────────────────────────────
def t_ontology_canonical():
    import disease_ontology as o
    assert len(o.CANONICAL_ORDER) == 21, len(o.CANONICAL_ORDER)
    nih = {o.to_canonical(n) for n in o.LABEL_SETS["NIH14"]}
    chex = {o.to_canonical(n) for n in o.LABEL_SETS["CHEXPERT14"]}
    assert len(nih & chex) == 7
    assert nih | chex == set(o.CANONICAL_ORDER)


def t_ontology_position_independence():
    """The bug this whole layer exists to prevent."""
    import disease_ontology as o
    v = [0.0] * 14
    v[3] = 0.9
    assert o.map_predictions("NIH14", v)["Infiltration"] == 0.9
    assert o.map_predictions("CHEXPERT14", v)["Lung Lesion"] == 0.9


def t_ontology_alias():
    import disease_ontology as o
    assert o.to_canonical("Pleural Effusion") == "Effusion"
    assert o.to_canonical("pleural_thickening") == "Pleural Thickening"


def t_registry_quarantine():
    """An unverified checkpoint must never auto-load."""
    import model_loader as ml
    try:
        ml.load_one("nih_densenet_laptop_ep20", "cpu")
        raise AssertionError("unverified checkpoint loaded — quarantine broken")
    except ValueError:
        pass


# ── symptom fusion ──────────────────────────────────────────────────────────
def t_fusion_no_evidence_no_change():
    """The bug that made a blank form produce a 96% urgent referral."""
    import symptom_fusion as sf
    r = sf.fuse_one("Pneumonia", 0.40, {})
    assert r["posterior"] == r["prior"], r


def t_fusion_direction():
    import symptom_fusion as sf
    up = sf.fuse_one("Pneumonia", 0.40,
                     {"fever_high": True, "cough_productive": True})
    dn = sf.fuse_one("Pneumonia", 0.40,
                     {"fever_high": False, "cough_productive": False})
    assert up["posterior"] > 0.40 < dn["posterior"] + 1  # up rises
    assert dn["posterior"] < 0.40, dn


def t_fusion_skips_non_pathology():
    import symptom_fusion as sf
    for d in ("Support Devices", "No Finding"):
        assert sf.fuse_one(d, 0.5, {"fever_high": True})["delta"] == 0.0


# ── stage 2 ─────────────────────────────────────────────────────────────────
def t_stage2_no_penalty_when_unanswered():
    """The bug that silently cut a 99% finding to 89.1%."""
    import stage2_questions as s2
    p = [{"disease": "Cardiomegaly", "probability": 0.99}]
    out = s2.apply_stage2_scores([dict(x) for x in p], {}, ["Cardiomegaly"])
    assert abs(out[0]["stage2_score"] - 0.99) < 1e-6, out[0]


def t_stage2_answers_move_score():
    import stage2_questions as s2
    p = [{"disease": "Pneumonia", "probability": 0.45}]
    yes = s2.apply_stage2_scores([dict(x) for x in p],
                                 {"pneu_fever": True, "pneu_sputum": True},
                                 ["Pneumonia"])[0]["stage2_score"]
    no = s2.apply_stage2_scores([dict(x) for x in p],
                                {"pneu_fever": False, "pneu_sputum": False},
                                ["Pneumonia"])[0]["stage2_score"]
    assert yes > 0.45 > no, (yes, no)


# ── knowledge base ──────────────────────────────────────────────────────────
def t_kb_covers_all_canonical():
    import medical_knowledge as mk, medical_knowledge_ext as ext
    import disease_ontology as o
    ext.install(mk.DISEASE_INFO, mk.DISEASE_CLINICAL_SOURCES)
    missing = [d for d in o.CANONICAL_ORDER if d not in mk.DISEASE_INFO]
    assert not missing, missing


def t_kb_schema_consistent():
    import medical_knowledge as mk, medical_knowledge_ext as ext
    ext.install(mk.DISEASE_INFO, mk.DISEASE_CLINICAL_SOURCES)
    ref = set(mk.DISEASE_INFO["Cardiomegaly"])
    for d, e in mk.DISEASE_INFO.items():
        assert set(e) == ref, f"{d} schema differs: {ref ^ set(e)}"


def t_kb_unknown_finding_safe():
    import medical_knowledge as mk, medical_knowledge_ext as ext
    e = ext.get_info_safe(mk.DISEASE_INFO, "Totally New Finding")
    assert "not in knowledge base" in e["full_name"]


# ── quality / calibration / extras ──────────────────────────────────────────
def t_quality_never_raises():
    import image_quality as iq
    assert iq.assess("not an image")["available"] is False
    assert iq.assess(None)["available"] is False


def t_quality_catches_damage():
    import numpy as np, image_quality as iq
    from PIL import Image
    g = np.random.RandomState(0).rand(256, 256).astype("float32") * 0.4 + 0.3
    dark = Image.fromarray((g * 0.15 * 255).astype("uint8"))
    assert iq.assess(dark)["overall"] in ("warn", "severe")


def t_calibration_is_monotonic():
    """Calibration must never reorder predictions — that would change AUC."""
    import calibration as cal
    d = "Pneumonia"
    xs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    ys = [cal.calibrate(d, x) for x in xs]
    assert all(a < b for a, b in zip(ys, ys[1:])), ys


def t_abstention_logic():
    import clinical_extras as cx
    assert cx.should_abstain(0.93, 0.02)["abstain"] is False
    assert cx.should_abstain(0.55, 0.20)["abstain"] is True


def t_zone_laterality():
    """Image-left is the patient's RIGHT. Getting this wrong reports findings
    on the wrong side of the body."""
    import numpy as np, clinical_extras as cx
    h = np.zeros((90, 90)); h[5:25, 5:25] = 1.0        # top-left of image
    z = cx.zone_of(h)
    assert z["row"] == "upper" and z["side"] == "right", z


def t_zone_flat_returns_none():
    import numpy as np, clinical_extras as cx
    assert cx.zone_of(np.ones((40, 40))) is None


# ── vindr labels ────────────────────────────────────────────────────────────
def t_vindr_labels_if_present():
    import csv
    p = Path("data/vindr/vindr_labels.csv")
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert len(rows) == 15000, len(rows)
    assert all(int(r["n_readers"]) == 3 for r in rows[:200])


# ── integration ─────────────────────────────────────────────────────────────
def t_app_boots():
    import importlib
    if "app" in sys.modules:
        del sys.modules["app"]
    app = importlib.import_module("app")
    assert app.film_model is None, "FiLM must stay retired"
    assert len(app.active_diseases) == 21, len(app.active_diseases)
    assert app.LOADED, "no models loaded"


def t_end_to_end_request():
    import importlib, json
    app = importlib.import_module("app")
    c = app.app.test_client()
    imgs = sorted(Path("data/vindr/train").glob("*.png"))[:1]
    if not imgs:
        return
    with imgs[0].open("rb") as f:
        r = c.post("/predict_stage1",
                   data={"image": (f, "x.png"), "age": "45",
                         "symptoms": json.dumps({"fever_high": True})},
                   content_type="multipart/form-data")
    assert r.status_code == 200, r.status_code
    j = r.get_json()
    assert j and not j.get("error"), j
    assert len(j["predictions"]) == 21
    assert j["calibrated"] is True
    assert "quality" in j and "study_verdict" in j
    assert j["heatmap"]


def t_no_false_claims_in_live_code():
    """Guards against the impossible-AUC and MICCAI claims reappearing."""
    src = Path("core/app.py").read_text(encoding="utf-8")
    for bad in ("0.3226", "0.9725", "MICCAI"):
        assert bad not in src, f"{bad!r} is back in core/app.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip tests that load model weights")
    a = ap.parse_args()

    print("=" * 70)
    print("  AyShCXR test suite")
    print("=" * 70)

    print("\nmodule self-tests")
    for m in ("disease_ontology", "symptom_fusion", "medical_knowledge_ext",
              "image_quality", "clinical_extras"):
        check(f"selftest:{m}", module_selftest(m))

    print("\nontology + registry")
    check("ontology: 21 canonical", t_ontology_canonical)
    check("ontology: position independence", t_ontology_position_independence)
    check("ontology: aliases", t_ontology_alias)
    check("registry: unverified quarantined", t_registry_quarantine)

    print("\nsymptom fusion")
    check("fusion: no evidence -> no change", t_fusion_no_evidence_no_change)
    check("fusion: moves both directions", t_fusion_direction)
    check("fusion: skips non-pathology", t_fusion_skips_non_pathology)

    print("\nstage 2")
    check("stage2: unanswered not penalised", t_stage2_no_penalty_when_unanswered)
    check("stage2: answers move score", t_stage2_answers_move_score)

    print("\nknowledge base")
    check("kb: covers all 21", t_kb_covers_all_canonical)
    check("kb: schema consistent", t_kb_schema_consistent)
    check("kb: unknown finding safe", t_kb_unknown_finding_safe)

    print("\nquality / calibration / extras")
    check("quality: never raises", t_quality_never_raises)
    check("quality: catches damage", t_quality_catches_damage)
    check("calibration: monotonic", t_calibration_is_monotonic)
    check("abstention: logic", t_abstention_logic)
    check("zones: laterality", t_zone_laterality)
    check("zones: flat -> none", t_zone_flat_returns_none)

    print("\ndata")
    check("vindr labels", t_vindr_labels_if_present)

    print("\nintegration")
    check("no false claims in live code", t_no_false_claims_in_live_code)
    check("app boots", t_app_boots, needs_models=True, fast=a.fast)
    check("end-to-end request", t_end_to_end_request, needs_models=True, fast=a.fast)

    print("\n" + "=" * 70)
    print(f"  PASS {len(PASS)}   FAIL {len(FAIL)}   SKIP {len(SKIP)}")
    if FAIL:
        print("\n  FAILURES:")
        for n, e in FAIL:
            print(f"    {n}\n      {e}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
