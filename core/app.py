# app.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi & Ayush Singh
# Two-Stage Clinical Decision System
#
# v6 Fixes Applied:
#   1. ✅ Explicit model loading — always loads 0.8031 safe model
#   2. ✅ MC Dropout passes increased 10 → 20 for stable uncertainty
#   3. ✅ EfficientNet GradCAM hook fixed features[6] → features[5]
#   4. ✅ X-ray image validation — rejects non-medical images
#   5. ✅ Per-disease optimal thresholds from calibration analysis
#   6. ✅ Disease dependency correction — fixes Infiltration overconfidence
#   7. ✅ GradCAM smoothed + correct DenseNet hook confirmed


# ── AyShCXR path bootstrap (added 2026-08-08 during folder reorganisation) ──
# This script now lives in a subfolder but still refers to data files by bare
# name (e.g. "nih_full_labels.csv"). Pointing the working directory at the
# project root keeps every existing relative path working unchanged, and puts
# core/ on sys.path so `import medical_knowledge` still resolves.
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
if _ROOT is None:
    raise RuntimeError(
        "Could not locate the AyShCXR project root: no .ayshcxr_root marker "
        f"found above {_HERE}. Restore that file or run from the project root."
    )
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Windows consoles frequently default to cp1252, which cannot encode the status
# emoji used below — startup would die with UnicodeEncodeError before the server
# ever came up. Force UTF-8 and degrade gracefully rather than crash.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# ── end path bootstrap ──────────────────────────────────────────────────────

import os
import io
import json
import base64
import torch
# torchvision `models` is still needed at the Grad-CAM hook, which checks
# isinstance(model, models.EfficientNet) to pick the right feature layer.
# `transforms` and `torch.nn` are no longer imported here: model_loader builds
# every transform from the registry, and the network definitions moved there too.
from torchvision import models
from PIL import Image, ImageStat
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string
from medical_knowledge import (
    get_disease_report, DISEASE_INFO, get_risk_summary
)
from stage2_questions import (
    get_stage2_questions_for_diseases, apply_stage2_scores
)
from symptom_fusion import fuse as bayes_fuse, explain as bayes_explain
from disease_ontology import CANONICAL_FINDINGS

app = Flask(__name__)

# ── Upload limits ───────────────────────────────────────────────────────────
# Without MAX_CONTENT_LENGTH, file.stream.read() buffers whatever arrives — a
# 2GB upload would be held entirely in memory. 40MB comfortably covers a
# full-resolution DICOM-derived PNG while making memory exhaustion impossible.
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

# Decompression-bomb guard. A small PNG can declare enormous dimensions and
# expand to gigabytes when decoded — a well-known attack against any service
# that opens user-supplied images. PIL warns above this and refuses above 2x it.
# 80 megapixels is far beyond any real radiograph (a 1024x1024 film is 1MP).
Image.MAX_IMAGE_PIXELS = 80_000_000


@app.errorhandler(413)
def _too_large(e):
    return jsonify({
        "error": "That file is too large (limit 40 MB). Chest radiographs are "
                 "normally well under 10 MB — check you are uploading the "
                 "X-ray and not a video or an archive."
    }), 413

# IMG_SIZE and `transform` are NOT set here any more. Each loaded model declares
# its own input size and normalisation in disease_ontology.MODEL_REGISTRY, and
# model_loader builds a matching transform per model:
#     CheXpert DenseNet-121 / EfficientNet-B4 -> 380px, 3-channel
#     Rad-DINO                                -> 224px, 3-channel
#     NIH DenseNet-121 (Phase 1)              -> 224px, 1-channel
# Those match the training scripts exactly; feeding a model the wrong resolution
# silently degrades it.
#
# IMG_SIZE and `transform` are deliberately NOT given placeholder values here.
# An earlier version set IMG_SIZE = None and built a 1-channel 224px transform
# at import time, both overwritten later by the registry. That looked harmless
# but was a trap: if registry loading ever failed, the app would carry on with a
# greyscale transform feeding 3-channel models and die deep inside Normalize
# with a shape error, far from the real cause. Leaving them undefined means any
# use before loading raises an immediate, obvious NameError instead.

# The old hardcoded 14-item DISEASES list has been removed. The finding list is
# now derived from whichever models are loaded — see `active_diseases`, built by
# model_loader.covered_findings() from the canonical 21 in disease_ontology.
# Keeping a second hardcoded list here guaranteed the two would drift apart.

# ── Per-disease decision thresholds ─────────────────────────────────────────
# Loaded from results/disease_thresholds.json, generated by
# reports/compute_thresholds.py.
#
# These are computed on the REAL pipeline (every loaded model -> merge ->
# calibrate) against radiologist-labelled VinDr images, NOT hand-picked and NOT
# taken from any single model's historical CSV. That distinction matters: the
# app never displays one model's raw output, so a threshold derived from one
# would not correspond to anything the user sees.
#
#   'measured' findings  -> F1-optimal against radiologist consensus
#   'baseline' findings  -> no ground truth available; set at the 90th
#                           percentile of ordinary films. Distributional, NOT
#                           validated — treat with caution.
def _load_thresholds():
    try:
        with open("results/disease_thresholds.json", encoding="utf-8") as f:
            data = json.load(f)
        th = data["thresholds"]
        print(f"✅ Thresholds: {len(th)} findings loaded "
              f"({sum(1 for v in data.get('basis', {}).values() if v == 'measured')} "
              f"measured against radiologist labels)")
        return th, data.get("basis", {})
    except Exception as e:
        print(f"⚠️  disease_thresholds.json unusable ({e}) — using 0.50/0.35 "
              f"for every finding. Run reports/compute_thresholds.py.")
        return {}, {}

DISEASE_THRESHOLDS, THRESHOLD_BASIS = _load_thresholds()

# The Python copy of STAGE1_SYMPTOM_CATEGORIES was removed on 2026-08-10.
# It was dead code: the browser builds the symptom checklist from its own
# STAGE1_CATEGORIES array inside the HTML/JS below, and this server-side
# duplicate was never read. Two copies of the same list guarantee they drift
# apart, and the JS one is the copy that actually reaches the user.

# ── Device ────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

# (no module-level `transform` — see the note above. Each model's transform is
#  built by model_loader from its registry entry; PRIMARY's is assigned to the
#  module-level `transform` during registry loading, for Grad-CAM.)

# ── X-Ray Validation ──────────────────────────────────
def _chest_signature(gray_small):
    """Score how much the image looks like a frontal chest radiograph.

    A chest X-ray has a distinctive horizontal profile through the mid-chest: a
    BRIGHT central band (spine and mediastinum, dense tissue absorbs X-rays)
    flanked by two DARK regions (air-filled lungs). Nothing else in a clinic
    looks like that — a hand, knee or dental film has no such structure.

    Returns roughly -0.5 (inverted / rotated) to +0.8 (textbook chest).
    Measured on 400 real radiographs: median +0.449, p1 +0.055, min -0.045.

    LIMITATION, stated plainly: this detects "not chest-shaped", not "definitely
    a chest". A synthetic image with a bright centre scores +0.657 and would
    pass. It reliably catches rotated (-0.192), inverted (-0.513), blank and
    noise images, which are the realistic failure modes in a PHC.
    """
    a = gray_small
    h, w = a.shape
    band = a[int(h * 0.25):int(h * 0.70), :]          # mid-chest rows only
    col = band.mean(axis=0)
    centre = col[int(w * 0.42):int(w * 0.58)].mean()  # mediastinum + spine
    lung_l = col[int(w * 0.15):int(w * 0.35)].mean()  # patient's RIGHT lung
    lung_r = col[int(w * 0.65):int(w * 0.85)].mean()  # patient's LEFT lung
    spread = float(col.max() - col.min())
    if spread < 1e-6:
        return 0.0
    return float((centre - (lung_l + lung_r) / 2.0) / spread)


def validate_xray(img_pil):
    """
    Validates that the uploaded image is likely a chest X-ray.
    Returns: (is_valid: bool, reason: str)

    Rejection messages tell the operator what to DO, not just what is wrong — a
    health worker seeing "aspect ratio unusual" has no idea what to try next.
    """
    img_gray = img_pil.convert("L")
    width, height = img_pil.size

    # ── Provenance of each limit ───────────────────────────────────────────
    # 500 real chest X-rays (VinDr + NIH) profiled on 2026-08-10:
    #   min dimension  512 – 1024   every one >= 512
    #   contrast std   28.4 – 91.2  limit 20  -> 0% rejected
    #   mean bright    77 – 197     limit 220 -> 0% rejected
    #   saturation     0.000        limit 0.15-> 0% rejected
    #   chest score    p1 +0.055    limit 0.0 -> <1% rejected
    #
    # HONEST CAVEAT on the aspect-ratio limit: every measured image was exactly
    # 1.00 because VinDr and NIH both ship pre-squared images. That measurement
    # therefore says NOTHING about real clinical films, which are typically
    # portrait (~4:5). The 0.5–2.0 window is a reasoned guess, not a validated
    # threshold, and it is deliberately wide for that reason.

    # Check 1: Minimum size.
    # Raised 100 -> 256. The models take 380px input, so a 150px upload gets
    # upscaled ~2.5x and every finding is inferred from invented pixels. Real
    # radiographs are >= 512px; anything under 256 is a thumbnail or a
    # screenshot fragment, and reading it confidently is worse than refusing.
    MIN_DIM = 256
    if width < MIN_DIM or height < MIN_DIM:
        return False, (
            f"Image is too small ({width}x{height}). A chest radiograph of at "
            f"least {MIN_DIM}x{MIN_DIM} pixels is required — smaller images must "
            f"be upscaled, which fabricates detail the AI would then read as "
            f"findings. Please upload the original file rather than a thumbnail."
        )

    # Check 2: Aspect ratio.
    # Deliberately wide — see the caveat above. This only catches obviously
    # non-radiographic shapes such as banners, panoramas or UI screenshots.
    ratio = width / height
    if ratio < 0.5 or ratio > 2.0:
        return False, (
            f"This image is {width}x{height}, which is far wider or taller than "
            f"a chest radiograph. It may be a screenshot, a photo of a screen, "
            f"or a cropped section. Please upload the full radiograph."
        )

    # Check 3: Colour. Merged from two previously separate checks (mean channel
    # difference, and per-pixel saturation) which measured the same property.
    # Saturation is the stronger of the two: a photo taken under mixed lighting
    # can have similar channel MEANS while still being visibly colour.
    pixels = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    maxc = pixels.max(axis=2)
    minc = pixels.min(axis=2)
    saturation = float(np.divide(maxc - minc, maxc,
                                 out=np.zeros_like(maxc), where=maxc > 0).mean())
    if saturation > 0.15:
        return False, (
            "This looks like a colour photograph rather than an X-ray. "
            "Chest radiographs are greyscale. If you photographed a printed "
            "film with a phone, retake it straight-on without coloured lighting, "
            "or upload the digital file from the X-ray machine instead."
        )

    stat_gray = ImageStat.Stat(img_gray)

    # Check 4: Contrast. A radiograph spans dense bright bone to dark air; a
    # blank, fogged or heavily compressed image does not.
    if stat_gray.stddev[0] < 20:
        return False, (
            "This image has almost no contrast between light and dark areas, so "
            "there is nothing for the AI to read. It may be blank, fogged, or "
            "saved at very low quality. Please upload the original file."
        )

    # Check 5: Gross overexposure.
    if stat_gray.mean[0] > 220:
        return False, (
            "This image is almost entirely white. If it is a photograph of a "
            "film on a lightbox, move away from the light source and retake it. "
            "If it came from the X-ray machine, the exposure settings need "
            "adjusting and the film should be repeated."
        )

    # Check 6: Chest-specific structure.
    # Everything above would happily accept a hand, knee or dental X-ray —
    # greyscale, high contrast, right shape, all checks pass — and the model
    # would then confidently report LUNG findings on a wrist. This is the only
    # check that asks whether the anatomy is actually a chest.
    small = np.asarray(img_gray.resize((256, 256)), dtype=np.float32) / 255.0
    score = _chest_signature(small)
    if score < 0.0:
        return False, (
            "This does not appear to be a frontal chest X-ray. The system looks "
            "for the bright central spine with darker lung fields either side, "
            "and could not find that pattern. Common causes: the image is "
            "rotated or sideways, the colours are inverted, or it is an X-ray "
            "of a different body part. Please check the orientation and that "
            "this is a chest (PA or AP) film."
        )

    return True, "Valid X-ray image"


# ── load_model() REMOVED — 2026-08-10 ──────────────────────────────────────
# 73 lines of dead code deleted. It was the original single-model loader that
# located checkpoints by guessing filenames (densenet121_BEST_auc0.8031...,
# then efficientnet_b4_14class_best.pth, then a fallback) and hardcoded one
# architecture, one input size and one 14-item label set.
#
# It has not been called since registry-driven loading replaced it below.
# Deleted rather than left in place because a second, stale loader sitting in
# the file is exactly the kind of thing that gets accidentally re-enabled.
# The old version is kept in the authors' private archive (not published).

# ── REGISTRY-DRIVEN LOADING — 2026-08-10 ────────────────────────────────────
# The old load_model() knew one architecture, one input size and one label set,
# and found checkpoints by guessing filenames. Every property a checkpoint needs
# now lives in disease_ontology.MODEL_REGISTRY, so adding a model later means
# one registry entry and no change here.
#
# ENSEMBLE_IDS is the only line to edit to change what the app serves.
#   - all four        -> best accuracy, 21 findings, needs internet on first run
#                        for Rad-DINO's HuggingFace download
#   - CNNs only       -> fully offline, still 21 findings
#   - single CheXpert -> the edge-deployment candidate
import model_loader as _ml
import medical_knowledge_ext as _mkext
import image_quality as _iq
import calibration as _calib
import clinical_extras as _cx
import risk_factors as _rf
import finding_reliability as _rel

# extend the knowledge base to all 21 canonical findings BEFORE anything can
# request a report for a CheXpert-only finding such as Fracture or Lung Opacity
from medical_knowledge import DISEASE_CLINICAL_SOURCES as _CLINICAL_SOURCES
_added = _mkext.install(DISEASE_INFO, _CLINICAL_SOURCES)
print(f"✅ Knowledge base extended: +{_added} findings (total {len(DISEASE_INFO)})")

ENSEMBLE_IDS = [m.strip() for m in os.environ.get(
    "AYSHCXR_MODELS",
    "chexpert_densenet121_v1,chexpert_efficientnet_b4_v1,"
    "chexpert_rad_dino_v1,nih_densenet121_v1"
).split(",") if m.strip()]

LOADED, _failed = _ml.load_models(ENSEMBLE_IDS, device)
for _mid, _why in _failed:
    print(f"⚠️  {_mid} unavailable — {_why}")

if not LOADED:
    # sys.exit(1), not exit(). exit() is a site builtin meant for the REPL, is
    # not guaranteed outside interactive use, and returns status 0 — a process
    # manager or Docker healthcheck would read "no models loaded" as a clean
    # shutdown and not restart.
    print("❌ No models could be loaded. Check models/ contains the .pth files, "
          "or that AYSHCXR_MODELS names entries that exist in MODEL_REGISTRY.")
    _sys.exit(1)

# Loading fewer models than requested is survivable but must never be silent:
# the app would run on one weak model while the log only said "Loaded 1 model(s)".
if len(LOADED) < len(ENSEMBLE_IDS):
    print(f"⚠️  DEGRADED: {len(ENSEMBLE_IDS)} model(s) requested, "
          f"{len(LOADED)} loaded. Predictions come from a reduced ensemble.")

# Grad-CAM needs a CNN with spatial feature maps; a transformer has none in the
# same form, so Rad-DINO cannot be the attribution model.
#
# Select the HIGHEST-AUC CNN explicitly. A previous version took the first
# non-Rad-DINO entry in list order and merely claimed it was the best — true
# only because chexpert_densenet121_v1 happens to lead ENSEMBLE_IDS. Reordering
# that string would have silently moved the heatmap to a weaker model, and the
# heatmap is the most visible output in the whole app.
_cnns = [e for e in LOADED if e["spec"]["arch"] != "rad_dino"]
if _cnns:
    PRIMARY = max(_cnns, key=lambda e: e["auc"] or 0.0)
else:
    # Only a transformer loaded. It can still predict; Grad-CAM will fail and is
    # caught at the call site, so the app degrades to "no heatmap" rather than
    # crashing.
    PRIMARY = LOADED[0]
    print("⚠️  No CNN loaded — Grad-CAM heatmaps will be unavailable.")

model = PRIMARY["model"]
transform = PRIMARY["transform"]
IMG_SIZE = PRIMARY["spec"]["img_size"]

active_diseases = _ml.covered_findings(LOADED)
num_diseases = len(active_diseases)

# findings only ONE model can predict carry less weight than a consensus of
# four — the UI flags these so a clinician knows which are single-source
_coverage_count = {}
for _e in LOADED:
    for _n in _ml.LABEL_SETS[_e["label_set"]]:
        _c = _ml.to_canonical(_n)
        _coverage_count[_c] = _coverage_count.get(_c, 0) + 1
SINGLE_SOURCE = {d for d, c in _coverage_count.items() if c == 1}

model_name = (f"Ensemble of {len(LOADED)}" if len(LOADED) > 1
              else PRIMARY["id"].replace("_", " "))
_best_auc = max((e["auc"] or 0) for e in LOADED)
model_name += f" (best member AUC {_best_auc:.4f})"

print(f"✅ Loaded {len(LOADED)} model(s): {[e['id'] for e in LOADED]}")
print(f"✅ Covering {num_diseases} findings "
      f"({len(SINGLE_SOURCE)} from a single model)")
print(f"✅ Grad-CAM source: {PRIMARY['id']}")
print(f"✅ Calibration: {'loaded' if _calib.load() else 'NOT FOUND — probabilities uncalibrated'}")


import torch.nn.functional as _F   # still used by GradCAM paths

# ── FiLM RETIRED — 2026-08-10 ───────────────────────────────────────────────
# The FiLM checkpoints (film_fusion_best_auc0.95-0.97*.pth) are NOT loaded and
# must never be re-enabled. Their reported AUC is invalid: the symptom vectors
# they trained on were GENERATED FROM THE DISEASE LABELS via
# SYMPTOM_PREVALENCE_BY_DISEASE, so the input already encoded the answer. Tested
# honestly with uninformative symptoms the same model scored 0.6878 — worse than
# the image alone.
#
# Symptoms are now handled by core/symptom_fusion.py: a Bayesian likelihood-ratio
# update with no training at all, so label leakage is structurally impossible.
# The .pth files are retained (never delete checkpoints) but are inert.
film_model = None
print("ℹ️  FiLM retired — symptoms handled by Bayesian likelihood-ratio fusion")

# ── GradCAM ───────────────────────────────────────────
# generate_gradcam() returns BOTH the coloured overlay and the raw heatmap.
#
# An earlier version stashed the raw map in a module-level dict (_LAST_CAM) for
# zone localisation to pick up afterwards. That works only because Flask's debug
# server is single-threaded: under any real WSGI server two users uploading at
# the same time would overwrite each other's heatmap, and patient A would be
# told the finding is in patient B's lung zone. Returning it makes the data flow
# explicit and the function safe to call concurrently.


def _primary_index(disease_name):
    """Canonical finding name -> index in PRIMARY model's OWN output vector.

    Necessary because predictions are now indexed by the 21 canonical findings
    while Grad-CAM must index the primary model's 14 raw outputs. Passing the
    canonical index straight through would attribute the heatmap to the wrong
    class — silently, and plausibly enough that nobody would notice.
    """
    labels = _ml.LABEL_SETS[PRIMARY["label_set"]]
    for i, raw in enumerate(labels):
        if _ml.to_canonical(raw) == disease_name:
            return i
    return None


def generate_gradcam(img_pil, target_class):
    """Grad-CAM attribution for one class.

    Returns (overlay_base64, raw_cam) — raw_cam is the un-coloured float map
    used by zone localisation. Returns (None, None) if attribution fails.

    NOTE: gradients are required here, so there is deliberately no
    torch.no_grad() around the forward pass.
    """
    # Use PRIMARY explicitly rather than the module-level `model` global. They
    # reference the same object today, but a function that hooks one model and
    # forward-passes another would be almost impossible to debug.
    net = PRIMARY["model"]
    tf = PRIMARY["transform"]
    size = PRIMARY["spec"]["img_size"]

    tensor = tf(img_pil).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    gradients, activations = [], []

    def save_grad(grad):   gradients.append(grad)
    def hook_fn(m, i, o):
        activations.append(o)
        o.register_hook(save_grad)

    # v6 FIX: EfficientNet hook corrected features[6] → features[5]
    if isinstance(net, models.EfficientNet):
        hook_layer = net.features[5]
    elif hasattr(net, "features") and hasattr(net.features, "denseblock4"):
        hook_layer = net.features.denseblock4      # confirmed correct for DenseNet
    elif hasattr(net, "layer4"):
        hook_layer = net.layer4
    elif hasattr(net, "features"):
        hook_layer = net.features[-1]
    else:
        # e.g. a bare transformer — no conv feature maps to attribute over
        print("⚠️  Grad-CAM: no suitable convolutional layer on "
              f"{type(net).__name__}")
        return None, None

    handle = hook_layer.register_forward_hook(hook_fn)
    try:
        net.eval()
        net.zero_grad(set_to_none=True)
        output = net(tensor)
        if target_class >= output.shape[1]:
            print(f"⚠️  Grad-CAM: class index {target_class} out of range for "
                  f"{output.shape[1]} outputs")
            return None, None
        output[0, target_class].backward()
    finally:
        # remove the hook even if the backward pass raised, otherwise every
        # failed request leaves another hook attached to the model
        handle.remove()

    if not gradients or not activations:
        print("⚠️  Grad-CAM: hooks captured nothing — no attribution produced")
        return None, None

    grad = gradients[0].squeeze().detach().cpu().numpy()
    act  = activations[0].squeeze().detach().cpu().numpy()

    if grad.ndim == 3 and act.ndim == 3:
        weights = np.mean(grad, axis=(1, 2))
        cam = np.tensordot(weights, act, axes=([0], [0])).astype(np.float32)
    else:
        cam = grad if grad.ndim == 2 else grad.mean(0)

    cam = np.maximum(cam, 0)

    # Correct min-max normalisation. The previous form divided by cam.max()
    # alone, which is only equivalent because ReLU guarantees cam.min() == 0 —
    # right by accident, and silently wrong for any other input. If the map is
    # completely flat there is nothing to show, so return a blank heatmap rather
    # than letting an unnormalised array reach np.uint8() as garbage.
    rng = float(cam.max() - cam.min())
    if rng > 1e-8:
        cam = (cam - cam.min()) / rng
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    # ORDER MATTERS: resize first, then blur.
    # The previous order blurred at feature-map resolution — roughly 12x12 for
    # DenseNet at 380px — where a 5x5 kernel spans almost half the map and
    # flattens the peak. Measured result: the heatmap maximum fell to 0.22-0.43,
    # so the JET colourmap never reached red and every overlay came out washed
    # out in green and yellow regardless of how confident the model was.
    # Upscaling first means the same kernel is a gentle smooth on a 380px map.
    cam = cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (11, 11), 0)

    # renormalise: cubic interpolation can overshoot slightly, and blurring
    # always shaves the peak. Without this the brightest region of a confident
    # prediction still would not render as red.
    rng2 = float(cam.max() - cam.min())
    if rng2 > 1e-8:
        cam = (cam - cam.min()) / rng2

    img_rgb = np.array(img_pil.resize((size, size)).convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)),
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)

    # release accumulated gradients — they are of no further use and otherwise
    # sit on the model between requests
    net.zero_grad(set_to_none=True)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), cam

# ── apply_disease_dependencies() REPLACED — 2026-08-10 ──────────────────────
# The old version applied four hand-written multiplicative rules, e.g. "if
# cardiac disease > 0.35 and Infiltration > 0.40, multiply Infiltration by 0.5".
# Three problems made it useless and slightly harmful:
#
#   1. Its thresholds (0.35 / 0.40 / 0.45 / 0.50 / 0.55) were tuned for the old
#      single NIH model. On the current merged+calibrated pipeline the medians
#      are Cardiomegaly 0.052, Edema 0.039, Consolidation 0.033 — so Cardiomegaly
#      had to reach SEVEN TIMES its median before a rule even considered firing.
#      Most rules were effectively dead.
#   2. Multiplicative boosts are not probability maths. x1.15 on 0.90 clamps at
#      0.99; on 0.05 it does almost nothing. Everything else in this system now
#      works in log-odds precisely to avoid that.
#   3. It patched one symptom (Infiltration looking too high) rather than the
#      cause: raw probabilities are not comparable ACROSS findings at all.
#
# Replaced by salience ranking in clinical_extras — see the measured A/B
# comparison in that file. Displayed probabilities are unchanged; only the
# ordering of findings changes. Measured effect on 333 single-finding
# radiologist-labelled films: top-1 accuracy 30% -> 39%, with Effusion 84% -> 91%
# and Pneumothorax 82% -> 91%.
#
# The clinical relationships the old rules encoded (cardiac disease explaining
# perihilar haze, consolidation supporting pneumonia) are real and worth
# revisiting — but as likelihood ratios in symptom_fusion, not as arbitrary
# multipliers on probabilities.

def load_finding_baselines():
    """Per-finding baseline distribution, from results/finding_baselines.json."""
    try:
        with open("results/finding_baselines.json", encoding="utf-8") as f:
            b = json.load(f)["baselines"]
        print(f"✅ Baselines: {len(b)} findings — ranking by salience")
        return b
    except Exception as e:
        print(f"⚠️  finding_baselines.json unusable ({e}) — falling back to raw "
              f"probability ranking, which buries findings with low typical "
              f"scores. Regenerate it to restore salience ranking.")
        return {}

FINDING_BASELINES = load_finding_baselines()

# ── History boost ─────────────────────────────────────
# _build_film_symptom_vector() removed 2026-08-10 — 57 lines that packed the
# 36 symptom flags into a tensor for the retired FiLM model. Nothing consumes
# it; symptom_fusion.py works from the plain dict instead.

def apply_symptom_fusion(img_pil, predictions, symptoms, smoking="no",
                         occupation="", conditions=""):
    """Bayesian fusion of clinical evidence with the image probability.

    ONE update, two evidence sources, both as likelihood ratios in log-odds:
      * reported SYMPTOMS      — symptom_fusion, from published prevalence data
      * history RISK FACTORS   — risk_factors, from occupation / comorbidities

    They are combined in a single pass rather than applied in sequence, which is
    what stops the same evidence being counted twice. Smoking, for example, is
    now handled once — as a risk factor — instead of being both injected into
    the symptom dict and separately boosted from the occupation string.

    Unlike the old 90/10 blend this can move a probability DOWN when the
    evidence argues against a finding, and weights each item by how
    discriminative it actually is rather than treating them alike.

    Each prediction gains `fusion` — the full audit trail — so the UI can show
    the clinician exactly which symptom moved the number and by how much.
    """
    # History as likelihood ratios. Word-boundary matched and negation-aware,
    # so "no diabetes" and "admitted" no longer fire the diabetes factor.
    risks = _rf.parse(occupation=occupation, conditions=conditions,
                      smoking=smoking)
    risk_lrs = _rf.likelihood_ratios(risks)

    # CRITICAL — tick-only semantics.
    # The Stage-1 checklist posts every one of the 36 keys, sending false for any
    # box the user did not tick. An untickedn box means "NOT REPORTED", not
    # "clinician has confirmed this symptom is absent". Those are different
    # clinical statements. Treating unticked as confirmed-absent applied ~30
    # negative likelihood ratios to a blank form and produced nonsense — a 46.7%
    # nodule became 96.1% with an urgent-referral banner on zero patient input.
    #
    # So Stage 1 counts POSITIVES ONLY. Explicit negatives are genuine evidence,
    # but they can only come from Stage 2, where the clinician is asked a direct
    # yes/no question and an answer of "no" really does mean absent.
    answered = {k: True for k, v in symptoms.items()
                if v is True or v == 1 or (isinstance(v, str) and v.lower() in ("true", "yes", "1"))}

    priors  = {p["disease"]: p["probability"] for p in predictions}
    results = bayes_fuse(priors, answered, risk_lrs=risk_lrs)

    fused = []
    for pred in predictions:
        r = results.get(pred["disease"])
        new_pred = dict(pred)
        if r and not r["skipped"]:
            new_pred["probability"]   = r["posterior"]
            new_pred["fusion_delta"]  = r["delta"]
            new_pred["fusion"]        = r["evidence"][:6]
            new_pred["fusion_prior"]  = r["prior"]
            new_pred["fusion_note"]   = bayes_explain(r)
            if r.get("risk_lr"):
                new_pred["risk_lr"] = r["risk_lr"]
                new_pred["risk_delta"] = r["risk_delta"]
        else:
            new_pred["fusion_delta"] = 0.0
            new_pred["fusion"]       = []
        fused.append(new_pred)
    # what history was detected, so the UI can show it and the clinician can
    # correct a mis-parse rather than wonder why a number moved
    return fused, [{"key": r["key"], "note": r["note"], "matched": r["matched"]}
                   for r in risks]

# ── apply_history_boost() REMOVED — 2026-08-10 ──────────────────────────────
# 34 lines that added a flat bonus to the probability for each risk factor
# (coal miner -> Fibrosis +0.08, hypertension -> Cardiomegaly +0.07, capped at
# +0.10). Four defects:
#
#   1. WRONG SCALE. Those constants assumed scores around 0.4-0.5. The current
#      pipeline has Cardiomegaly median 0.052 and Edema 0.039, so typing
#      "hypertension" MORE THAN DOUBLED Cardiomegaly before the image was even
#      weighed. Patient history was overwhelming the radiograph.
#   2. ADDITIVE, while every other evidence path works in log-odds.
#   3. DOUBLE COUNTING. It boosted Effusion for symptoms["swelling"] while
#      symptom_fusion applied a likelihood ratio to that same swelling. Also
#      tb_contact, night_sweats, haemoptysis, recent_surgery and fever.
#   4. NAIVE TEXT MATCHING. `"dm" in cond` matched "aDMitted" and "oeDeMa";
#      `"sugar"` matched "no sugar"; and "no history of diabetes" fired the
#      diabetes boost because negation was never checked.
#
# Replaced by core/risk_factors.py: word-boundary matching, negation-aware, and
# applied as likelihood ratios through the SAME Bayesian update as symptoms —
# so evidence combines correctly and nothing is counted twice. Symptoms are
# deliberately excluded from the risk table for that reason.

# ── MC Dropout ────────────────────────────────────────
def predict_with_uncertainty(img_pil, n_passes=20):
    """
    v6 FIX: n_passes increased from 10 → 20 for more stable uncertainty
    estimates. More passes = less variance in the uncertainty score itself.

    2026-08-10: now runs EVERY loaded model and merges by canonical finding
    name, then applies per-disease temperature calibration.

    Returns arrays aligned to `active_diseases` (21 canonical findings), so the
    rest of the app is unchanged.
    """
    merged, per_model, unc = _ml.predict(LOADED, img_pil, device,
                                         mc_passes=n_passes)

    # calibration makes a displayed "70%" mean 70%. Measured ECE fell from
    # 0.129 to 0.015 on the held-out test split. Monotonic, so ranking and AUC
    # are provably unchanged — it corrects meaning, not ordering.
    merged = _calib.calibrate_all(merged)

    # active_diseases is built from the loaded models' label sets, so every
    # entry should be present. The default exists only to survive a registry
    # inconsistency — and it is asserted rather than silently filled, because
    # a missing finding rendered as 0.0 is indistinguishable from "confidently
    # absent", which is a different and much stronger clinical statement.
    missing = [d for d in active_diseases if d not in merged]
    if missing:
        print(f"⚠️  {len(missing)} finding(s) absent from model output and "
              f"reported as 0.0: {missing[:5]}")

    mean_probs = np.array([merged.get(d, 0.0) for d in active_diseases],
                          dtype=np.float32)
    std_probs = np.array([unc.get(d, 0.0) for d in active_diseases],
                         dtype=np.float32)
    return mean_probs, std_probs

# ── HTML (full interface) ─────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AyShCXR — AI Chest X-Ray Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Instrument+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#060e1f;--surface:#091220;--glass:rgba(14,165,233,0.04);
  --glass-border:rgba(14,165,233,0.12);--accent:#38bdf8;--accent2:#0ea5e9;
  --gold:#f59e0b;--gold2:#b45309;--danger:#f43f5e;--warn:#fb923c;
  --success:#34d399;--purple:#818cf8;--purple2:#a5b4fc;
  --text:#e8f4fd;--muted:#6b7e95;--dim:#111d2e;
  --mono:'JetBrains Mono',monospace;--sans:'Instrument Sans',sans-serif;
  --display:'Syne',sans-serif;
  --card-bg:rgba(9,20,38,0.85);--card-border:rgba(56,189,248,0.12);
  --medical-blue:#0369a1;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:linear-gradient(150deg,#060e1f 0%,#071526 45%,#08111e 100%);background-attachment:fixed;color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden;}
.orb{position:fixed;border-radius:50%;filter:blur(140px);pointer-events:none;z-index:0;opacity:0.10;animation:orbFloat 10s ease-in-out infinite;}
.orb-1{width:700px;height:700px;background:radial-gradient(circle,#0ea5e9,#0369a1);top:-250px;left:-200px;}
.orb-2{width:500px;height:500px;background:radial-gradient(circle,#6366f1,#312e81);bottom:-150px;right:-100px;animation-delay:-5s;}
.orb-3{width:250px;height:250px;background:#0ea5e9;top:45%;left:42%;animation-delay:-3s;opacity:0.05;}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1);}50%{transform:translate(25px,15px) scale(1.04);}}
.grid-overlay{position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);background-size:60px 60px;z-index:0;pointer-events:none;}
.scanline{position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);z-index:0;pointer-events:none;}
.app-wrap{position:relative;z-index:1;min-height:100vh;}

/* HEADER */
header{padding:16px 40px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(0,212,255,0.1);background:rgba(6,8,18,0.85);backdrop-filter:blur(20px);position:sticky;top:0;z-index:100;}
.logo-wrap{display:flex;align-items:center;gap:14px;}
.logo-icon{width:44px;height:44px;background:linear-gradient(135deg,var(--accent),var(--purple));border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 24px rgba(0,212,255,0.3);}
.logo-text h1{font-family:var(--display);font-size:20px;font-weight:800;letter-spacing:4px;background:linear-gradient(135deg,var(--accent),var(--purple2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.logo-text p{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:2px;}
.header-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.auc-badge{display:flex;align-items:center;gap:8px;background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:20px;padding:5px 12px;}
.auc-dot{width:6px;height:6px;border-radius:50%;background:var(--success);box-shadow:0 0 8px var(--success);animation:pulse 2s ease infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
.auc-text{font-family:var(--mono);font-size:10px;color:var(--accent);letter-spacing:1px;}
.hdr-pill{font-family:var(--mono);font-size:10px;padding:5px 12px;border-radius:20px;border:1px solid var(--glass-border);background:var(--glass);color:var(--muted);letter-spacing:1px;}

/* HINDI TOGGLE */
.lang-toggle{display:flex;align-items:center;gap:8px;background:var(--glass);border:1px solid var(--glass-border);border-radius:20px;padding:4px 6px;cursor:pointer;transition:all 0.2s;}
.lang-btn{padding:4px 10px;border-radius:14px;font-family:var(--mono);font-size:10px;cursor:pointer;transition:all 0.2s;border:none;background:transparent;color:var(--muted);}
.lang-btn.active{background:rgba(0,212,255,0.15);color:var(--accent);border:1px solid rgba(0,212,255,0.3);}

/* MAIN */
main{max-width:1300px;margin:0 auto;padding:50px 24px;}

/* HERO */
.hero{text-align:center;margin-bottom:70px;}
.hero-tag{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10px;color:var(--accent);letter-spacing:2px;text-transform:uppercase;padding:6px 16px;border:1px solid rgba(56,189,248,0.25);border-radius:20px;background:rgba(56,189,248,0.06);margin-bottom:24px;box-shadow:0 0 20px rgba(56,189,248,0.06);}
.hero-tag::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);}
.hero h2{font-family:var(--display);font-size:60px;font-weight:800;line-height:1.05;margin-bottom:20px;background:linear-gradient(135deg,#e0f2fe 0%,#38bdf8 45%,#818cf8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero p{font-size:15px;color:var(--muted);max-width:600px;margin:0 auto 20px;line-height:1.8;font-weight:300;}
.hero-badges{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:32px;}
.hero-badge{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:9px;letter-spacing:1px;padding:5px 13px;border-radius:20px;border:1px solid;text-transform:uppercase;}
.hero-badge.blue{color:#38bdf8;border-color:rgba(56,189,248,0.25);background:rgba(56,189,248,0.06);}
.hero-badge.green{color:#34d399;border-color:rgba(52,211,153,0.25);background:rgba(52,211,153,0.06);}
.hero-badge.purple{color:#a5b4fc;border-color:rgba(165,180,252,0.25);background:rgba(165,180,252,0.06);}

/* STAGE FLOW */
.stage-flow{display:flex;align-items:center;justify-content:center;gap:0;margin-bottom:16px;}
.stage-node{display:flex;flex-direction:column;align-items:center;gap:8px;}
.stage-circle{width:46px;height:46px;border-radius:50%;border:2px solid var(--glass-border);background:var(--glass);display:flex;align-items:center;justify-content:center;font-family:var(--display);font-size:15px;font-weight:700;color:var(--muted);transition:all 0.4s ease;}
.stage-circle.active{border-color:var(--accent);background:rgba(0,212,255,0.1);color:var(--accent);box-shadow:0 0 20px rgba(0,212,255,0.3);}
.stage-circle.complete{border-color:var(--success);background:rgba(46,213,115,0.1);color:var(--success);box-shadow:0 0 20px rgba(46,213,115,0.3);}
.stage-label{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;text-align:center;max-width:80px;}
.stage-label.active{color:var(--accent);}
.stage-label.complete{color:var(--success);}
.stage-connector{width:80px;height:1px;background:var(--glass-border);margin:0 4px;margin-bottom:26px;}

/* DEMO MODE BANNER */
.demo-banner{background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.2);border-radius:16px;padding:16px 24px;margin-bottom:28px;display:flex;align-items:center;justify-content:space-between;gap:16px;animation:slideUp 0.4s ease;}
.demo-banner h4{font-family:var(--display);font-size:15px;font-weight:700;color:var(--gold);margin-bottom:4px;}
.demo-banner p{font-size:12px;color:var(--muted);}
.demo-cases{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}
.demo-case{padding:7px 14px;border-radius:8px;border:1px solid rgba(0,212,255,0.2);background:rgba(0,212,255,0.04);font-family:var(--mono);font-size:10px;color:var(--accent);cursor:pointer;transition:all 0.2s;}
.demo-case:hover{background:rgba(0,212,255,0.12);border-color:rgba(0,212,255,0.4);}
.close-demo{background:var(--glass);border:1px solid var(--glass-border);border-radius:8px;padding:6px 12px;font-family:var(--mono);font-size:10px;color:var(--muted);cursor:pointer;transition:all 0.2s;flex-shrink:0;}
.close-demo:hover{color:var(--text);}

/* CARDS */
.gcard{background:var(--card-bg);border:1px solid var(--card-border);border-radius:20px;padding:28px;backdrop-filter:blur(24px);position:relative;overflow:hidden;transition:transform 0.3s ease,box-shadow 0.3s ease;box-shadow:0 4px 32px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04);}
.gcard::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.2),transparent);}
.gcard:hover{transform:translateY(-2px);box-shadow:0 24px 64px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.05);}
.gcard.glow-cyan{border-color:rgba(56,189,248,0.2);box-shadow:0 4px 32px rgba(0,0,0,0.4),0 0 40px rgba(56,189,248,0.06),inset 0 1px 0 rgba(56,189,248,0.08);}
.gcard.glow-purple{border-color:rgba(129,140,248,0.2);box-shadow:0 4px 32px rgba(0,0,0,0.4),0 0 40px rgba(129,140,248,0.06),inset 0 1px 0 rgba(129,140,248,0.08);}
.ctitle{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:2.5px;color:var(--accent);margin-bottom:20px;display:flex;align-items:center;gap:10px;}
.ctitle-bar{width:3px;height:14px;background:linear-gradient(180deg,var(--accent),var(--purple));border-radius:2px;}

/* UPLOAD */
.upload-actions{display:flex;gap:10px;margin-bottom:12px;}
.upload-btn{flex:1;padding:10px;border-radius:10px;border:1px dashed rgba(0,212,255,0.3);background:rgba(0,212,255,0.04);color:var(--accent);font-family:var(--mono);font-size:11px;cursor:pointer;transition:all 0.2s;text-align:center;letter-spacing:1px;}
.upload-btn:hover{background:rgba(0,212,255,0.1);border-color:rgba(0,212,255,0.5);}
.upload-btn.camera{border-color:rgba(124,58,237,0.3);background:rgba(124,58,237,0.04);color:var(--purple2);}
.upload-btn.camera:hover{background:rgba(124,58,237,0.1);border-color:rgba(124,58,237,0.5);}
.upload-zone{border:2px dashed rgba(0,212,255,0.15);border-radius:16px;padding:32px 20px;text-align:center;cursor:pointer;transition:all 0.3s ease;position:relative;overflow:hidden;background:rgba(0,212,255,0.02);}
.upload-zone:hover,.upload-zone.dragging{border-color:var(--accent);background:rgba(0,212,255,0.04);}
.upload-zone.invalid{border-color:var(--danger);background:rgba(255,71,87,0.04);}
.upload-icon-wrap{width:64px;height:64px;margin:0 auto 14px;border-radius:50%;background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);display:flex;align-items:center;justify-content:center;font-size:28px;transition:all 0.3s ease;}
.upload-zone:hover .upload-icon-wrap{background:rgba(0,212,255,0.15);box-shadow:0 0 30px rgba(0,212,255,0.2);transform:scale(1.05);}
.upload-zone h3{font-family:var(--display);font-size:15px;font-weight:600;margin-bottom:5px;}
.upload-zone p{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:1px;}
#file-input,#camera-input{display:none;}
#preview-img{width:100%;max-height:220px;object-fit:contain;border-radius:10px;display:none;margin-top:14px;border:1px solid var(--glass-border);}
.validation-error{background:rgba(255,71,87,0.08);border:1px solid rgba(255,71,87,0.3);border-radius:10px;padding:10px 14px;font-size:12px;color:#ff6b7a;margin-top:12px;display:none;font-family:var(--mono);line-height:1.5;}

/* PREPROCESS STATUS */
.preprocess-status{display:none;margin-top:12px;padding:10px 14px;background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:10px;font-family:var(--mono);font-size:11px;color:var(--accent);}

/* FIELDS */
.field{margin-bottom:14px;}
.field label{display:block;font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;}
.field input,.field select,.field textarea{width:100%;background:rgba(255,255,255,0.03);border:1px solid var(--glass-border);border-radius:10px;padding:10px 14px;color:var(--text);font-family:var(--sans);font-size:13px;outline:none;transition:all 0.2s ease;}
.field input:focus,.field select:focus,.field textarea:focus{border-color:rgba(0,212,255,0.4);background:rgba(0,212,255,0.04);box-shadow:0 0 0 3px rgba(0,212,255,0.08);}
.field select option{background:#0d1117;}
.field textarea{resize:vertical;min-height:60px;}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.symptom-cat{display:flex;align-items:center;gap:10px;margin:22px 0 11px;}
.symptom-cat-icon{font-size:16px;width:30px;height:30px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.15);flex-shrink:0;}
.symptom-cat-text{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:1.8px;font-weight:700;color:#e2e8f0;}
.symptom-cat-line{flex:1;height:1px;background:linear-gradient(to right,rgba(56,189,248,0.2),transparent);}


.symptoms-grid{display:flex;flex-wrap:wrap;gap:7px;align-items:flex-start;}
.symptom-item{display:inline-flex;align-items:center;gap:6px;padding:6px 14px 6px 10px;border-radius:50px;cursor:pointer;border:1.5px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.035);color:#7a8fa8;font-size:12px;font-weight:500;user-select:none;transition:border-color 0.16s,background 0.16s,color 0.16s,box-shadow 0.16s;}
.symptom-item input[type="checkbox"]{display:none;}
.symptom-item::before{content:"";width:7px;height:7px;border-radius:50%;flex-shrink:0;background:rgba(255,255,255,0.12);transition:background 0.16s,box-shadow 0.16s;}
.symptom-item:hover{border-color:rgba(56,189,248,0.4);background:rgba(56,189,248,0.07);color:#cbd5e1;}
.symptom-item:hover::before{background:rgba(56,189,248,0.55);}
.symptom-item.sub-item{font-size:11px;padding:5px 13px 5px 10px;background:rgba(99,102,241,0.04);border-color:rgba(129,140,248,0.1);color:#5a6a82;}
.symptom-item.sub-item::before{background:rgba(129,140,248,0.2);}
.symptom-item.sub-item:hover{border-color:rgba(129,140,248,0.4);background:rgba(99,102,241,0.1);color:#a5b4fc;}
.symptom-item.checked{border-color:rgba(56,189,248,0.75);background:linear-gradient(135deg,rgba(56,189,248,0.16),rgba(56,189,248,0.07));color:#38bdf8;font-weight:600;box-shadow:0 0 14px rgba(56,189,248,0.2),inset 0 1px 0 rgba(255,255,255,0.07);}
.symptom-item.checked::before{background:#38bdf8;box-shadow:0 0 7px rgba(56,189,248,0.9);}
.symptom-item.sub-item.checked{border-color:rgba(167,139,250,0.75);background:linear-gradient(135deg,rgba(129,140,248,0.18),rgba(99,102,241,0.07));color:#c4b5fd;font-weight:600;box-shadow:0 0 14px rgba(129,140,248,0.18),inset 0 1px 0 rgba(255,255,255,0.07);}
.symptom-item.sub-item.checked::before{background:#a78bfa;box-shadow:0 0 7px rgba(167,139,250,0.9);}
.dur-row{width:100%;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 14px;border-radius:12px;background:rgba(99,102,241,0.04);border:1px solid rgba(129,140,248,0.1);margin-top:4px;}
.dur-row-label{font-size:10px;color:#4e5d70;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap;}

.dur-pill{display:inline-flex;align-items:center;padding:5px 14px;border:1.5px solid rgba(129,140,248,0.15);border-radius:50px;cursor:pointer;font-size:11px;font-weight:500;color:#5a6a82;background:rgba(9,18,36,0.5);transition:all 0.16s ease;user-select:none;}

.dur-pill:hover{border-color:rgba(129,140,248,0.5);color:#c4b5fd;background:rgba(99,102,241,0.1);}
.dur-pill.selected{border-color:rgba(167,139,250,0.85);background:linear-gradient(135deg,rgba(129,140,248,0.28),rgba(99,102,241,0.14));color:#e9d5ff;font-weight:700;box-shadow:0 0 12px rgba(129,140,248,0.22);}
.symp-total{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:14px;text-align:right;padding-top:8px;border-top:1px solid rgba(56,189,248,0.08);}


.symptom-count-badge{background:rgba(56,189,248,0.1);color:var(--accent);border:1px solid rgba(56,189,248,0.25);border-radius:10px;padding:1px 7px;font-size:9px;font-family:var(--mono);display:none;}
.symptom-count-badge.visible{display:inline;}









.dur-pills{display:flex;gap:6px;flex-wrap:wrap;}

.dur-pill input[type="radio"]{display:none;}


.film-delta{font-family:var(--mono);font-size:9px;padding:1px 5px;border-radius:4px;margin-left:4px;}
.film-delta.pos{color:var(--success);background:rgba(52,211,153,0.1);}
.film-delta.neg{color:var(--danger);background:rgba(244,63,94,0.1);}
.pre-film-track{height:2px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;margin-top:3px;}
.pre-film-fill{height:100%;border-radius:4px;background:rgba(255,255,255,0.15);}

/* DURATION SLIDER */
.dur-wrap{background:rgba(56,189,248,0.03);border:1px solid rgba(56,189,248,0.12);border-radius:12px;padding:16px 18px;margin-bottom:18px;}
.dur-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
.dur-label-txt{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);}
.dur-val-badge{font-family:var(--mono);font-size:10px;color:var(--accent);background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.2);border-radius:10px;padding:2px 10px;}
.dur-range{width:100%;-webkit-appearance:none;appearance:none;height:4px;border-radius:4px;outline:none;cursor:pointer;background:rgba(255,255,255,0.08);transition:background 0.2s;}
.dur-range::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;background:linear-gradient(135deg,var(--accent),var(--purple));border-radius:50%;cursor:pointer;box-shadow:0 0 14px rgba(56,189,248,0.5);border:2px solid rgba(255,255,255,0.25);transition:transform 0.15s ease;}
.dur-range::-webkit-slider-thumb:hover{transform:scale(1.2);}
.dur-steps{display:flex;justify-content:space-between;margin-top:8px;}
.dur-steps span{font-family:var(--mono);font-size:8px;color:var(--muted);text-align:center;flex:1;}

/* FORM GRID */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:32px;}

/* SUBMIT */
.submit-btn{width:100%;padding:16px;background:linear-gradient(135deg,var(--accent),var(--purple));color:#fff;border:none;border-radius:12px;font-family:var(--display);font-size:15px;font-weight:700;letter-spacing:1px;cursor:pointer;transition:all 0.3s ease;margin-top:20px;position:relative;overflow:hidden;}
.submit-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,transparent,rgba(255,255,255,0.1),transparent);transform:translateX(-100%);transition:transform 0.5s ease;}
.submit-btn:hover::before{transform:translateX(100%);}
.submit-btn:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(0,212,255,0.4);}
.submit-btn:disabled{opacity:0.4;cursor:not-allowed;transform:none;box-shadow:none;}

/* LOADING — IMPROVED */
.loading{display:none;text-align:center;padding:60px 20px;}
.loader-ring{width:80px;height:80px;margin:0 auto 28px;position:relative;}
.loader-ring::before,.loader-ring::after{content:'';position:absolute;inset:0;border-radius:50%;border:3px solid transparent;}
.loader-ring::before{border-top-color:var(--accent);animation:spin 1s linear infinite;}
.loader-ring::after{border-bottom-color:var(--purple);animation:spin 1.5s linear infinite reverse;}
@keyframes spin{to{transform:rotate(360deg);}}
.loader-inner{position:absolute;inset:12px;border-radius:50%;background:radial-gradient(circle,rgba(0,212,255,0.1),transparent);display:flex;align-items:center;justify-content:center;font-size:22px;}
.loading-msg{font-family:var(--mono);font-size:13px;color:var(--text);margin-bottom:20px;}

/* STEP PROGRESS */
.step-progress{max-width:400px;margin:0 auto;}
.step-item{display:flex;align-items:center;gap:12px;padding:8px 0;opacity:0.3;transition:opacity 0.3s ease;}
.step-item.active{opacity:1;}
.step-item.done{opacity:0.6;}
.step-dot{width:8px;height:8px;border-radius:50%;background:var(--glass-border);flex-shrink:0;transition:all 0.3s ease;}
.step-item.active .step-dot{background:var(--accent);box-shadow:0 0 10px var(--accent);}
.step-item.done .step-dot{background:var(--success);}
.step-label{font-family:var(--mono);font-size:11px;color:var(--muted);}
.step-item.active .step-label{color:var(--accent);}
.step-item.done .step-label{color:var(--success);}
.step-check{margin-left:auto;font-size:12px;opacity:0;}
.step-item.done .step-check{opacity:1;}

/* STAGE 2 */
#stage2-section{display:none;margin-top:40px;animation:slideUp 0.5s ease;}
@keyframes slideUp{from{opacity:0;transform:translateY(30px);}to{opacity:1;transform:translateY(0);}}
.s2-card{background:rgba(124,58,237,0.04);border:1px solid rgba(124,58,237,0.2);border-radius:20px;padding:36px;position:relative;overflow:hidden;}
.s2-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--purple),var(--accent),var(--purple));}
.s2-title{font-family:var(--display);font-size:24px;font-weight:800;background:linear-gradient(135deg,var(--purple2),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px;}
.s2-sub{font-size:13px;color:var(--muted);margin-bottom:24px;line-height:1.6;}
.disease-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px;}
.dchip{padding:6px 16px;border-radius:20px;font-family:var(--mono);font-size:11px;background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);color:var(--purple2);}
.s2-progress-bar{height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:6px;overflow:hidden;}
.s2-progress-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--accent));border-radius:2px;transition:width 0.4s ease;}
.s2-progress-text{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:20px;}
.s2-questions{display:flex;flex-direction:column;gap:12px;}
.s2-q{background:rgba(255,255,255,0.02);border:1px solid rgba(124,58,237,0.15);border-radius:12px;padding:16px 18px;transition:border-color 0.2s ease;}
.s2-q:hover{border-color:rgba(124,58,237,0.3);}
.s2-q-tag{font-family:var(--mono);font-size:9px;color:var(--purple2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
.s2-q-text{font-size:13px;color:var(--text);margin-bottom:12px;line-height:1.5;}
.yn-row{display:flex;gap:8px;}
.yn-btn{padding:7px 22px;border-radius:8px;font-family:var(--mono);font-size:11px;cursor:pointer;border:1px solid;background:transparent;transition:all 0.2s ease;}
.yn-btn.yes{border-color:rgba(46,213,115,0.4);color:var(--success);}
.yn-btn.yes:hover,.yn-btn.yes.selected{background:rgba(46,213,115,0.12);}
.yn-btn.no{border-color:rgba(255,71,87,0.4);color:var(--danger);}
.yn-btn.no:hover,.yn-btn.no.selected{background:rgba(255,71,87,0.12);}
.yn-btn.selected{font-weight:600;}
.s2-submit{width:100%;padding:16px;background:linear-gradient(135deg,var(--purple),var(--accent));color:#fff;border:none;border-radius:12px;font-family:var(--display);font-size:15px;font-weight:700;cursor:pointer;transition:all 0.3s ease;margin-top:24px;letter-spacing:1px;}
.s2-submit:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(124,58,237,0.4);}
.s2-submit:disabled{opacity:0.35;cursor:not-allowed;transform:none;}
.s2-note{text-align:center;font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:10px;}

/* ═══ TRAFFIC LIGHT ═══ */
.traffic-light-card{border-radius:20px;padding:32px;margin-bottom:32px;text-align:center;position:relative;overflow:hidden;animation:slideUp 0.5s ease;}
.traffic-light-card.tl-red{background:rgba(255,71,87,0.08);border:2px solid rgba(255,71,87,0.4);}
.traffic-light-card.tl-amber{background:rgba(255,165,2,0.08);border:2px solid rgba(255,165,2,0.4);}
.traffic-light-card.tl-green{background:rgba(46,213,115,0.08);border:2px solid rgba(46,213,115,0.4);}
.tl-lights{display:flex;justify-content:center;gap:16px;margin-bottom:20px;}
.tl-light{width:40px;height:40px;border-radius:50%;opacity:0.2;transition:all 0.5s ease;box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);}
.tl-light.red-bulb{background:#ff4757;}
.tl-light.amber-bulb{background:#ffa502;}
.tl-light.green-bulb{background:#2ed573;}
.tl-light.on{opacity:1;}
.tl-light.on.red-bulb{box-shadow:0 0 30px rgba(255,71,87,0.8),0 0 60px rgba(255,71,87,0.4);}
.tl-light.on.amber-bulb{box-shadow:0 0 30px rgba(255,165,2,0.8),0 0 60px rgba(255,165,2,0.4);}
.tl-light.on.green-bulb{box-shadow:0 0 30px rgba(46,213,115,0.8),0 0 60px rgba(46,213,115,0.4);}
.tl-action{font-family:var(--display);font-size:26px;font-weight:800;margin-bottom:8px;}
.tl-action.red{color:#ff4757;}
.tl-action.amber{color:#ffa502;}
.tl-action.green{color:#2ed573;}
.tl-detail{font-size:14px;color:var(--muted);margin-bottom:16px;line-height:1.5;}
.tl-disease{font-family:var(--mono);font-size:12px;padding:6px 16px;border-radius:20px;display:inline-block;}
.tl-disease.red{background:rgba(255,71,87,0.15);color:#ff4757;border:1px solid rgba(255,71,87,0.3);}
.tl-disease.amber{background:rgba(255,165,2,0.15);color:#ffa502;border:1px solid rgba(255,165,2,0.3);}
.tl-disease.green{background:rgba(46,213,115,0.15);color:#2ed573;border:1px solid rgba(46,213,115,0.3);}

/* ═══ CONFIDENCE METER ═══ */
.confidence-section{margin-bottom:28px;}
.confidence-wrap{display:flex;align-items:center;justify-content:center;gap:40px;flex-wrap:wrap;}
.gauge-wrap{position:relative;width:180px;height:100px;}
.gauge-svg{width:180px;height:100px;}
.gauge-label{position:absolute;bottom:0;left:50%;transform:translateX(-50%);text-align:center;}
.gauge-pct{font-family:var(--display);font-size:28px;font-weight:800;}
.gauge-sub{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;}
.confidence-detail{max-width:220px;}
.conf-title{font-family:var(--display);font-size:16px;font-weight:700;margin-bottom:8px;}
.conf-desc{font-size:12px;color:var(--muted);line-height:1.6;}
.conf-warn{background:rgba(255,165,2,0.08);border:1px solid rgba(255,165,2,0.2);border-radius:8px;padding:8px 12px;font-size:11px;color:#c8904a;margin-top:10px;font-family:var(--mono);}

/* ═══ RESULTS LAYOUT ═══ */
#results{display:none;margin-top:48px;animation:slideUp 0.5s ease;}

/* IMAGES */
.img-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px;}
.img-panel{border-radius:16px;overflow:hidden;border:1px solid var(--glass-border);background:var(--glass);position:relative;}
.img-panel-label{position:absolute;top:10px;left:10px;font-family:var(--mono);font-size:9px;letter-spacing:1.5px;text-transform:uppercase;padding:4px 10px;border-radius:20px;background:rgba(6,8,18,0.8);border:1px solid var(--glass-border);color:var(--muted);backdrop-filter:blur(10px);z-index:2;}
.img-panel img{width:100%;max-height:260px;object-fit:contain;display:block;}
.heatmap-legend{padding:8px 12px;display:flex;align-items:center;justify-content:center;gap:8px;font-family:var(--mono);font-size:9px;color:var(--muted);}
.legend-bar{width:80px;height:6px;border-radius:3px;background:linear-gradient(90deg,#0000ff,#00ff00,#ff0000);}

/* PRIMARY FINDING */
.primary-card{background:rgba(0,212,255,0.03);border:1px solid rgba(0,212,255,0.15);border-radius:20px;padding:28px;margin-bottom:24px;position:relative;overflow:hidden;}
.primary-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--purple),var(--accent));}
.primary-tag{font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin-bottom:10px;display:flex;align-items:center;gap:8px;}
.primary-tag::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent);animation:pulse 2s ease infinite;}
.primary-name{font-family:var(--display);font-size:32px;font-weight:800;color:var(--text);margin-bottom:4px;}
.primary-icd{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:20px;}
.score-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px;}
.score-block{flex:1;min-width:90px;background:rgba(255,255,255,0.03);border:1px solid var(--glass-border);border-radius:12px;padding:12px;text-align:center;}
.score-val{font-family:var(--mono);font-size:26px;font-weight:500;line-height:1;margin-bottom:5px;}
.score-val.c{color:var(--accent);}
.score-val.p{color:var(--purple2);}
.score-val.g{color:var(--success);}
.score-lbl{font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}
.primary-action{background:rgba(255,71,87,0.06);border:1px solid rgba(255,71,87,0.2);border-radius:10px;padding:10px 14px;font-size:13px;color:#ff6b7a;}

/* ═══ RADAR CHART ═══ */
.viz-tabs{display:flex;gap:8px;margin-bottom:16px;}
.viz-tab{padding:7px 16px;border-radius:8px;font-family:var(--mono);font-size:10px;cursor:pointer;border:1px solid var(--glass-border);background:var(--glass);color:var(--muted);transition:all 0.2s;}
.viz-tab.active{border-color:rgba(0,212,255,0.4);background:rgba(0,212,255,0.08);color:var(--accent);}
#radar-view{display:none;justify-content:center;align-items:center;padding:16px 0;}
#bars-view{display:block;}
.radar-wrap{position:relative;}
#radarCanvas{display:block;}
.radar-legend{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-top:12px;}
.radar-leg-item{display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:9px;color:var(--muted);}
.radar-leg-dot{width:8px;height:8px;border-radius:50%;}

/* BARS */
.bars-legend{display:flex;flex-direction:column;gap:5px;padding:10px 14px;margin-bottom:14px;background:rgba(56,189,248,0.03);border:1px solid rgba(56,189,248,0.09);border-radius:10px;}
.bl-item{font-size:10px;color:#6b7a90;display:flex;align-items:center;gap:7px;line-height:1.4;}
.bl-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.disease-row{margin-bottom:12px;}
.disease-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;font-size:12px;}
.disease-name{color:var(--text);display:flex;align-items:center;gap:6px;}
.disease-right{display:flex;align-items:center;gap:8px;}
.disease-pct{font-family:var(--mono);font-size:11px;min-width:42px;text-align:right;}
.unc-range{font-family:var(--mono);font-size:9px;color:var(--muted);}
.bar-track{height:4px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;transition:width 1.2s cubic-bezier(0.4,0,0.2,1);}
.bar-primary{background:linear-gradient(90deg,var(--accent),var(--purple));}
.bar-detected{background:var(--danger);}
.bar-warn{background:var(--warn);}
.bar-clear{background:rgba(255,255,255,0.1);}
.dbadge{font-size:9px;padding:2px 7px;border-radius:10px;font-family:var(--mono);}
.dbadge-primary{background:rgba(0,212,255,0.12);color:var(--accent);border:1px solid rgba(0,212,255,0.3);}
.dbadge-danger{background:rgba(255,71,87,0.12);color:var(--danger);border:1px solid rgba(255,71,87,0.3);}
.dbadge-warn{background:rgba(255,165,2,0.12);color:var(--warn);border:1px solid rgba(255,165,2,0.3);}

/* RISK / EXTRA */
.risk-box{background:rgba(245,166,35,0.04);border:1px solid rgba(245,166,35,0.15);border-left:3px solid var(--gold);border-radius:12px;padding:16px 18px;margin-bottom:20px;}
.risk-box h4{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--gold);margin-bottom:10px;}
.risk-box ul{list-style:none;}
.risk-box ul li{font-size:12px;color:#c8924a;padding:4px 0;display:flex;gap:8px;line-height:1.5;}
.risk-box ul li::before{content:'⚠';flex-shrink:0;}
.extra-box{background:rgba(255,71,87,0.04);border:1px solid rgba(255,71,87,0.15);border-left:3px solid var(--danger);border-radius:12px;padding:16px 18px;margin-bottom:20px;}
.extra-box h4{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--danger);margin-bottom:10px;}
.extra-chip{display:inline-flex;align-items:center;gap:6px;background:rgba(255,71,87,0.08);border:1px solid rgba(255,71,87,0.2);border-radius:8px;padding:5px 12px;font-size:11px;color:#ff6b7a;margin:3px;font-family:var(--mono);}

/* FILM PROOF PANEL */
.film-proof{background:rgba(52,211,153,0.03);border:1px solid rgba(52,211,153,0.18);border-radius:16px;padding:20px 24px;margin-bottom:20px;animation:slideUp 0.4s ease;}
.film-proof-hdr{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:6px;}
.film-proof-title{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:2px;color:var(--success);display:flex;align-items:center;gap:8px;}
.film-proof-sub{font-size:11px;color:var(--muted);margin-bottom:16px;line-height:1.6;}
.film-proof-auc-badge{font-family:var(--mono);font-size:10px;border-radius:10px;padding:3px 12px;white-space:nowrap;}
.film-proof-auc-badge.pos-badge{color:var(--success);background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.25);}
.film-proof-auc-badge.neg-badge{color:#f87171;background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.25);}
/* How it works strip */
.film-how-it-works{display:flex;align-items:flex-start;gap:0;margin:14px 0 14px;background:rgba(56,189,248,0.04);border:1px solid rgba(56,189,248,0.1);border-radius:12px;padding:14px 16px;flex-wrap:wrap;}
.film-hiw-step{flex:1;min-width:160px;display:flex;gap:10px;align-items:flex-start;font-size:11px;color:#94a3b8;line-height:1.55;}
.film-hiw-num{width:22px;height:22px;border-radius:50%;background:rgba(56,189,248,0.15);border:1px solid rgba(56,189,248,0.35);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;font-weight:700;color:var(--accent);flex-shrink:0;margin-top:1px;}
.film-hiw-arrow{align-self:center;color:rgba(56,189,248,0.4);font-size:18px;padding:0 8px;font-weight:300;}
/* legend dots */
.film-legend-dot{display:inline-block;width:8px;height:8px;border-radius:50%;vertical-align:middle;margin-right:3px;}
.grey-dot{background:rgba(255,255,255,0.25);}
.teal-dot{background:linear-gradient(90deg,#38bdf8,#34d399);}
.fpr-row{display:grid;grid-template-columns:120px 1fr 52px;align-items:center;gap:10px;margin-bottom:9px;}
.fpr-disease{font-size:11px;color:var(--text);}
.fpr-bars{display:flex;flex-direction:column;gap:4px;}
.fpr-bar-row{display:flex;align-items:center;gap:8px;}
.fpr-label{font-family:var(--mono);font-size:8px;color:var(--muted);width:82px;flex-shrink:0;}
.fpr-track{flex:1;height:5px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;}
.fpr-fill-pre{height:100%;border-radius:4px;background:rgba(255,255,255,0.18);transition:width 1s ease;}
.fpr-fill-post{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--success));transition:width 1.3s ease;}
.fpr-pct{font-family:var(--mono);font-size:8px;color:var(--muted);width:32px;text-align:right;flex-shrink:0;}
.fpr-delta{font-family:var(--mono);font-size:10px;font-weight:700;text-align:right;}
.fpr-delta.pos{color:var(--success);}
.fpr-delta.neg{color:var(--danger);}
.film-proof-divider{height:1px;background:rgba(52,211,153,0.1);margin:12px 0;}
.film-proof-footer{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:9px;color:var(--muted);}
/* REPORTS */
.report-card{border-radius:16px;padding:24px;margin-bottom:20px;border-left:3px solid;background:rgba(255,255,255,0.02);}
.report-card.rp{border-color:var(--accent);background:rgba(0,212,255,0.03);}
.report-card.rh{border-color:var(--danger);background:rgba(255,71,87,0.03);}
.report-card.rm{border-color:var(--warn);background:rgba(255,165,2,0.03);}
.report-card.rl{border-color:var(--success);background:rgba(46,213,115,0.03);}
.report-card.rx{border-color:var(--gold);background:rgba(245,166,35,0.03);}
.r-dname{font-family:var(--display);font-size:20px;font-weight:700;margin-bottom:2px;}
.r-icd{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:6px;}
.r-prob{font-family:var(--mono);font-size:12px;margin-bottom:14px;color:var(--muted);}
.urgency-pill{display:inline-block;padding:4px 14px;border-radius:20px;font-size:9px;font-family:var(--mono);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:18px;}
.up-primary{background:rgba(0,212,255,0.12);color:var(--accent);border:1px solid rgba(0,212,255,0.3);}
.up-high{background:rgba(255,71,87,0.12);color:var(--danger);border:1px solid rgba(255,71,87,0.3);}
.up-medium{background:rgba(255,165,2,0.12);color:var(--warn);border:1px solid rgba(255,165,2,0.3);}
.up-low{background:rgba(46,213,115,0.12);color:var(--success);border:1px solid rgba(46,213,115,0.3);}
.r-section{margin-bottom:14px;}
.r-section h5{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:6px;}
.r-section p{font-size:13px;line-height:1.6;color:#94a3b8;}
.r-section ul{list-style:none;}
.r-section ul li{font-size:12px;color:#94a3b8;padding:2px 0;display:flex;align-items:flex-start;gap:8px;line-height:1.5;}
.r-section ul li::before{content:'→';color:var(--accent);flex-shrink:0;}
.specialist-tag{display:inline-flex;align-items:center;gap:8px;background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.2);border-radius:8px;padding:8px 14px;font-size:13px;color:var(--gold);}
.tag-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;}
.rtag{background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.2);border-radius:6px;padding:3px 10px;font-size:10px;color:var(--gold);font-family:var(--mono);}
.labtag{background:rgba(46,213,115,0.06);border:1px solid rgba(46,213,115,0.2);border-radius:6px;padding:3px 10px;font-size:10px;color:var(--success);font-family:var(--mono);}
.emergency-box{background:rgba(255,71,87,0.06);border:1px solid rgba(255,71,87,0.2);border-radius:8px;padding:10px 14px;font-size:12px;color:#ff6b7a;line-height:1.5;margin-top:14px;}
.followup-box{background:rgba(245,166,35,0.04);border:1px solid rgba(245,166,35,0.15);border-radius:8px;padding:10px 14px;font-size:12px;color:#c8904a;line-height:1.5;margin-top:8px;}
.history-label{display:inline-block;font-family:var(--mono);font-size:9px;padding:3px 10px;border-radius:20px;background:rgba(245,166,35,0.1);color:var(--gold);border:1px solid rgba(245,166,35,0.3);margin-bottom:14px;}
.separator{text-align:center;padding:20px 0;font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase;}
.of-chip{background:var(--glass);border:1px solid var(--glass-border);border-radius:10px;padding:10px 16px;font-size:12px;}
.of-name{color:var(--text);font-weight:500;margin-bottom:2px;}
.of-pct{font-family:var(--mono);font-size:10px;color:var(--muted);}
.other-findings{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;}
.disclaimer{background:rgba(245,166,35,0.04);border:1px solid rgba(245,166,35,0.15);border-radius:12px;padding:18px 24px;font-size:12px;color:#c8904a;text-align:center;margin-top:40px;line-height:1.7;font-family:var(--mono);}
.action-row{display:flex;gap:12px;justify-content:center;margin-top:28px;flex-wrap:wrap;}
.act-btn{padding:11px 24px;border-radius:10px;cursor:pointer;font-size:13px;font-family:var(--sans);font-weight:500;transition:all 0.2s ease;}
.act-btn.export{background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.25);color:var(--accent);}
.act-btn.export:hover{background:rgba(0,212,255,0.15);}
.act-btn.clear{background:var(--glass);border:1px solid var(--glass-border);color:var(--muted);}
.act-btn.clear:hover{color:var(--text);}
.act-btn.pdf{background:rgba(46,213,115,0.08);border:1px solid rgba(46,213,115,0.25);color:var(--success);}
.act-btn.pdf:hover{background:rgba(46,213,115,0.15);}

/* HINDI translations shown/hidden */
[data-en],[data-hi]{transition:opacity 0.2s ease;}
.hindi-mode [data-en]{display:none!important;}
.hindi-mode [data-hi]{display:block!important;}
[data-hi]{display:none;}

@media(max-width:768px){
  .form-grid,.img-grid,.two-col,.symptoms-grid{grid-template-columns:1fr;}
  header{padding:12px 16px;flex-wrap:wrap;gap:8px;}
  main{padding:28px 16px;}
  .hero h2{font-size:38px;}
  .stage-connector{width:32px;}
  .score-row{flex-wrap:wrap;}
}
</style>
</head>
<body>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
<div class="grid-overlay"></div>
<div class="scanline"></div>

<div class="app-wrap" id="app-root">

<!-- HEADER -->
<header>
  <div class="logo-wrap">
    <div class="logo-icon">🫁</div>
    <div class="logo-text">
      <h1>AyShCXR</h1>
      <p>AI Chest X-Ray Analysis System</p>
    </div>
  </div>
  <div class="header-right">
    <span style="font-family:var(--mono);font-size:11px;color:var(--muted)">Subhrakant Sethi &amp; Ayush Singh</span>
    <div class="auc-badge">
      <div class="auc-dot"></div>
      <span class="auc-text" id="model-badge">Loading...</span>
    </div>
    <!-- HINDI TOGGLE -->
    <div class="lang-toggle" onclick="toggleLang()">
      <button class="lang-btn active" id="btn-en">EN</button>
      <button class="lang-btn" id="btn-hi">हिं</button>
    </div>
    <!-- DEMO MODE -->
    <button class="act-btn export" style="padding:5px 12px;font-size:10px" onclick="showDemo()">🎬 Demo</button>
    <span class="hdr-pill" id="disease-count-pill">Loading…</span>
  </div>
</header>

<main>

  <!-- HERO -->
  <div class="hero">
    <div class="hero-tag">Clinical AI Research System · TIET 2026</div>
    <h2 data-en>AI-Assisted<br>Chest Screening</h2>
    <h2 data-hi style="display:none">AI-सहायक<br>छाती की जांच</h2>
    <p data-en>Two-stage AI for rural PHC deployment — DenseNet-121 image analysis combined with patient symptoms via Bayesian likelihood ratios drawn from published clinical guidelines. Narrows 14 pathologies to one primary diagnosis with GradCAM spatial evidence.</p>
    <p data-hi style="display:none">ग्रामीण PHC के लिए दो-चरण AI — छाती की 14 बीमारियों का विश्लेषण, नैदानिक दिशानिर्देशों पर आधारित बायेसियन लक्षण संलयन के साथ।</p>
    <div class="hero-badges">
      <span class="hero-badge blue">🏥 NIH ChestX-ray14 · AUC 0.8031</span>
      <span class="hero-badge green">✦ Bayesian Symptom Fusion · Guideline-Sourced</span>
      <span class="hero-badge purple">🩺 Explainable · GradCAM + Uncertainty</span>
    </div>
    <div class="stage-flow">
      <div class="stage-node">
        <div class="stage-circle active" id="pill-1">1</div>
        <div class="stage-label active" id="lbl-1" data-en>Broad Screening</div>
      </div>
      <div class="stage-connector" id="con-1"></div>
      <div class="stage-node">
        <div class="stage-circle" id="pill-2">2</div>
        <div class="stage-label" id="lbl-2" data-en>Targeted Narrowing</div>
      </div>
      <div class="stage-connector" id="con-2"></div>
      <div class="stage-node">
        <div class="stage-circle" id="pill-3">3</div>
        <div class="stage-label" id="lbl-3" data-en>Primary Diagnosis</div>
      </div>
    </div>
  </div>

  <!-- DEMO MODE PANEL -->
  <div id="demo-panel" style="display:none">
    <div class="demo-banner">
      <div>
        <h4>🎬 Demo Mode — Conference Presentation</h4>
        <p>Select a pre-loaded clinical case. Each demonstrates the full two-stage diagnostic flow.</p>
        <div class="demo-cases">
          <div class="demo-case" onclick="loadDemo('cardiomegaly')">❤️ Cardiomegaly</div>
          <div class="demo-case" onclick="loadDemo('pneumothorax')">💨 Pneumothorax</div>
          <div class="demo-case" onclick="loadDemo('effusion')">💧 Pleural Effusion</div>
          <div class="demo-case" onclick="loadDemo('emphysema')">🫧 Emphysema</div>
          <div class="demo-case" onclick="loadDemo('pneumonia')">🦠 Pneumonia</div>
          <div class="demo-case" onclick="loadDemo('fibrosis')">🧱 Fibrosis</div>
          <div class="demo-case" onclick="loadDemo('mass')">⚠️ Pulmonary Mass</div>
          <div class="demo-case" onclick="loadDemo('edema')">🌊 Pulmonary Edema</div>
          <div class="demo-case" onclick="loadDemo('nodule')">🔴 Nodule</div>
          <div class="demo-case" onclick="loadDemo('atelectasis')">📉 Atelectasis</div>
        </div>
      </div>
      <button class="close-demo" onclick="hideDemo()">✕ Close</button>
    </div>
  </div>

  <!-- FORM GRID -->
  <div class="form-grid">
    <div>
      <!-- UPLOAD -->
      <div class="gcard glow-cyan" style="margin-bottom:20px">
        <div class="ctitle"><div class="ctitle-bar"></div><span data-en>Chest X-Ray Image</span><span data-hi style="display:none">छाती X-Ray छवि</span></div>
        <!-- Camera + Upload buttons -->
        <div class="upload-actions">
          <div class="upload-btn" onclick="document.getElementById('file-input').click()">
            📁 <span data-en>Upload File</span><span data-hi style="display:none">फ़ाइल अपलोड</span>
          </div>
          <div class="upload-btn camera" onclick="document.getElementById('camera-input').click()">
            📸 <span data-en>Use Camera</span><span data-hi style="display:none">कैमरा</span>
          </div>
        </div>
        <div class="upload-zone" id="upload-zone"
             onclick="document.getElementById('file-input').click()"
             ondragover="handleDrag(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)">
          <div class="upload-icon-wrap" id="upload-icon">📡</div>
          <h3 id="upload-h3" data-en>Upload or Capture Chest X-Ray</h3>
          <p id="upload-p" data-en>Chest radiograph only · PNG JPG · PA or AP view · Drag &amp; drop supported</p>
        </div>
        <input type="file" id="file-input" accept=".png,.jpg,.jpeg" onchange="handleFile(this)">
        <input type="file" id="camera-input" accept="image/*" capture="environment" onchange="handleCameraFile(this)">
        <img id="preview-img" alt="Preview">
        <div class="preprocess-status" id="preprocess-status">
          🔧 <span data-en>Phone photo detected — applying X-ray enhancement preprocessing...</span>
          <span data-hi style="display:none">फोन फोटो पहचाना गया — X-Ray सुधार लागू हो रहा है...</span>
        </div>
        <div class="validation-error" id="validation-error"></div>
      </div>

      <!-- PATIENT INFO -->
      <div class="gcard">
        <div class="ctitle"><div class="ctitle-bar"></div><span data-en>Patient Information</span><span data-hi style="display:none">रोगी की जानकारी</span></div>
        <div class="field"><label data-en>Full Name / Initials</label><label data-hi style="display:none">पूरा नाम</label><input type="text" id="pt-name" placeholder="e.g. Subhrakant S."></div>
        <div class="field"><label data-en>Patient ID (optional)</label><label data-hi style="display:none">रोगी ID</label><input type="text" id="pt-id" placeholder="Hospital / clinic ID"></div>
        <div class="two-col">
          <div class="field"><label data-en>Age</label><label data-hi style="display:none">आयु</label><input type="number" id="pt-age" placeholder="25" min="1" max="120"></div>
          <div class="field"><label data-en>Gender</label><label data-hi style="display:none">लिंग</label>
            <select id="pt-gender">
              <option value="Male">Male / पुरुष</option>
              <option value="Female">Female / महिला</option>
              <option value="Other">Other / अन्य</option>
            </select>
          </div>
        </div>
        <div class="field"><label data-en>Symptom Duration</label><label data-hi style="display:none">लक्षण की अवधि</label><input type="text" id="pt-duration" placeholder="e.g. 3 days / 3 दिन"></div>
        <div class="two-col">
          <div class="field"><label data-en>Smoking History</label><label data-hi style="display:none">धूम्रपान</label>
            <select id="pt-smoking">
              <option value="no">Non-smoker / धूम्रपान नहीं</option>
              <option value="yes">Current smoker / धूम्रपान करता है</option>
              <option value="past">Ex-smoker / पूर्व धूम्रपायी</option>
            </select>
          </div>
          <div class="field"><label data-en>Occupation</label><label data-hi style="display:none">पेशा</label><input type="text" id="pt-occupation" placeholder="e.g. Coal Miner / कोयला खान"></div>
        </div>
        <div class="field"><label data-en>Known Medical Conditions</label><label data-hi style="display:none">ज्ञात बीमारियां</label><input type="text" id="pt-conditions" placeholder="e.g. Diabetes / मधुमेह"></div>
        <div class="field"><label data-en>Current Medications</label><label data-hi style="display:none">वर्तमान दवाएं</label><input type="text" id="pt-medications" placeholder="e.g. Amiodarone, Steroids"></div>
        <div class="field"><label data-en>Referring Doctor</label><label data-hi style="display:none">रेफर करने वाले डॉक्टर</label><input type="text" id="pt-doctor" placeholder="Dr. Name"></div>
        <div class="field"><label data-en>Additional Notes</label><label data-hi style="display:none">अतिरिक्त नोट</label><textarea id="pt-notes" placeholder="Any other relevant information / अन्य जानकारी"></textarea></div>
      </div>
    </div>

    <!-- STAGE 1 SYMPTOMS -->
    <div class="gcard glow-purple">
      <div class="ctitle"><div class="ctitle-bar"></div><span data-en>Stage 1 — Symptom Checklist</span><span data-hi style="display:none">चरण 1 — लक्षण सूची</span></div>
      <p style="font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.7">
        <span data-en>Select all symptoms currently present. Grade severity where shown. These 36 clinical dimensions (incl. symptom duration) update each probability by its published likelihood ratio.</span>
        <span data-hi style="display:none">सभी मौजूदा लक्षण चुनें। ये 36 नैदानिक आयाम प्रकाशित संभावना अनुपात के आधार पर प्रत्येक संभावना को अद्यतन करते हैं।</span>
      </p>
      <!-- SYMPTOM DURATION SLIDER -->
      <div class="dur-wrap">
        <div class="dur-hdr">
          <span class="dur-label-txt">📅 &nbsp;How long have these symptoms been present?</span>
          <span class="dur-val-badge" id="dur-badge">Not specified</span>
        </div>
        <input type="range" id="sym-dur-range" min="0" max="4" step="1" value="0"
               class="dur-range" oninput="updateDuration(this.value)">
        <div class="dur-steps">
          <span>&lt; 1 wk</span><span>1–2 wks</span><span>2–4 wks</span><span>1–3 mo</span><span>&gt; 3 mo</span>
        </div>
      </div>
      <div id="symptoms-container"></div>
      <button class="submit-btn" id="analyse-btn" onclick="runStage1()">
        🔬 &nbsp;<span data-en>Analyse X-Ray — Stage 1</span><span data-hi style="display:none">X-Ray विश्लेषण — चरण 1</span>
      </button>
    </div>
  </div>

  <!-- STAGE 2 -->
  <div id="stage2-section">
    <div class="s2-card">
      <div class="s2-title" data-en>Stage 2 — Targeted Clinical Verification</div>
      <div class="s2-title" data-hi style="display:none">चरण 2 — लक्षित नैदानिक सत्यापन</div>
      <div class="s2-sub" data-en>Answer all questions to narrow to one primary diagnosis. All questions are mandatory.</div>
      <div class="s2-sub" data-hi style="display:none">एक प्राथमिक निदान तक पहुंचने के लिए सभी प्रश्नों का उत्तर दें।</div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px" data-en>Diseases Under Consideration</div>
      <div class="disease-chips" id="stage2-top-diseases"></div>
      <div class="s2-progress-text" id="stage2-progress-text">0 of 0 questions answered</div>
      <div class="s2-progress-bar"><div class="s2-progress-fill" id="stage2-progress-fill" style="width:0%"></div></div>
      <div class="s2-questions" id="stage2-questions-container"></div>
      <button class="s2-submit" id="stage2-submit-btn" onclick="runStage2()" disabled>
        🎯 &nbsp;<span data-en>Find Primary Diagnosis — Stage 2</span><span data-hi style="display:none">प्राथमिक निदान खोजें — चरण 2</span>
      </button>
      <div class="s2-note" data-en>All questions must be answered to proceed</div>
      <div class="s2-note" data-hi style="display:none">आगे बढ़ने के लिए सभी प्रश्नों का उत्तर देना अनिवार्य है</div>
    </div>
  </div>

  <!-- LOADING — IMPROVED STEP PROGRESS -->
  <div class="loading" id="loading">
    <div class="loader-ring"><div class="loader-inner">🫁</div></div>
    <div class="loading-msg" id="loading-msg">Initialising...</div>
    <div class="step-progress" id="step-progress">
      <div class="step-item" id="step-1"><div class="step-dot"></div><span class="step-label" data-en>Validating X-ray image</span><span class="step-label" data-hi style="display:none">X-Ray छवि सत्यापन</span><span class="step-check">✓</span></div>
      <div class="step-item" id="step-2"><div class="step-dot"></div><span class="step-label" data-en>Running neural network inference</span><span class="step-label" data-hi style="display:none">तंत्रिका नेटवर्क चला रहा है</span><span class="step-check">✓</span></div>
      <div class="step-item" id="step-3"><div class="step-dot"></div><span class="step-label" data-en>Analysing 14 disease patterns</span><span class="step-label" data-hi style="display:none">14 बीमारी पैटर्न विश्लेषण</span><span class="step-check">✓</span></div>
      <div class="step-item" id="step-4"><div class="step-dot"></div><span class="step-label" data-en>Generating GradCAM heatmap</span><span class="step-label" data-hi style="display:none">GradCAM हीटमैप बना रहा है</span><span class="step-check">✓</span></div>
      <div class="step-item" id="step-5"><div class="step-dot"></div><span class="step-label" data-en>Compiling clinical report</span><span class="step-label" data-hi style="display:none">नैदानिक रिपोर्ट संकलित</span><span class="step-check">✓</span></div>
    </div>
  </div>

  <!-- RESULTS -->
  <div id="results">

    <!-- TRAFFIC LIGHT — TOP OF RESULTS -->
    <div id="traffic-light-box"></div>

    <!-- IMAGES SIDE BY SIDE -->
    <div class="img-grid">
      <div class="img-panel">
        <div class="img-panel-label" data-en>Original X-Ray</div>
        <div class="img-panel-label" data-hi style="display:none">मूल X-Ray</div>
        <img id="result-xray" src="" alt="">
      </div>
      <div class="img-panel">
        <div class="img-panel-label" id="heatmap-label" data-en>AI Attention Heatmap</div>
        <div class="img-panel-label" data-hi style="display:none">AI ध्यान हीटमैप</div>
        <img id="result-heatmap" src="" alt="">
        <div class="heatmap-legend">
          <span data-en>Low</span><span data-hi style="display:none">कम</span>
          <div class="legend-bar"></div>
          <span data-en>High Attention</span><span data-hi style="display:none">उच्च ध्यान</span>
        </div>
      </div>
    </div>

    <!-- CONFIDENCE METER + PRIMARY FINDING -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px" id="confidence-primary-grid">
      <div class="gcard" id="confidence-section">
        <div class="ctitle"><div class="ctitle-bar"></div><span data-en>Confidence Meter</span><span data-hi style="display:none">विश्वास मीटर</span></div>
        <div class="confidence-wrap">
          <div class="gauge-wrap">
            <canvas id="gaugeCanvas" width="180" height="100"></canvas>
            <div class="gauge-label">
              <div class="gauge-pct" id="gauge-pct">--</div>
              <div class="gauge-sub" data-en>Confidence</div>
              <div class="gauge-sub" data-hi style="display:none">विश्वास</div>
            </div>
          </div>
          <div class="confidence-detail">
            <div class="conf-title" id="conf-title">--</div>
            <div class="conf-desc" id="conf-desc"></div>
            <div class="conf-warn" id="conf-warn" style="display:none"></div>
          </div>
        </div>
      </div>
      <div id="primary-finding-box" style="display:none"></div>
    </div>

    <!-- OTHER FINDINGS -->
    <div id="other-findings-box" style="display:none">
      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:12px" data-en>Other Considered Findings</div>
      <div class="other-findings" id="other-findings-list"></div>
    </div>

    <!-- RISK NOTES -->
    <div id="risk-summary-box" style="display:none" class="risk-box">
      <h4 data-en>⚠ Clinical Risk Factors From Patient History</h4>
      <h4 data-hi style="display:none">⚠ रोगी इतिहास से नैदानिक जोखिम कारक</h4>
      <ul id="risk-summary-list"></ul>
    </div>

    <!-- EXTRA DISEASES -->
    <div id="extra-diseases-box" style="display:none" class="extra-box">
      <h4 data-en>🔍 Additionally Suspected From Patient History</h4>
      <h4 data-hi style="display:none">🔍 रोगी इतिहास से अतिरिक्त संदिग्ध</h4>
      <div id="extra-diseases-list"></div>
    </div>

    <!-- PROBABILITY VISUALISATION — BARS + RADAR -->
    <div class="gcard" style="margin-bottom:24px">
      <div class="ctitle"><div class="ctitle-bar"></div>
        <span data-en>Top Findings · Differential Diagnosis</span>
        <span data-hi style="display:none">शीर्ष निष्कर्ष · विभेदक निदान</span>
        <span style="font-size:9px;color:var(--muted);margin-left:4px">(primary + considered · expand to see all)</span>
      </div>
      <div class="viz-tabs">
        <div class="viz-tab active" onclick="switchViz('bars',this)" data-en>Bar Chart</div>
        <div class="viz-tab" onclick="switchViz('radar',this)" data-en>Radar Chart</div>
      </div>
      <div id="bars-view">
        <div class="bars-legend">
          <span class="bl-item"><span class="bl-dot" style="background:var(--accent)"></span>Bar = AI confidence score (0–100%) that the disease is present in this X-ray</span>
          <span class="bl-item"><span class="bl-dot" style="background:rgba(255,255,255,0.2)"></span>Thin ghost bar = DenseNet-121 score <em>before</em> symptoms applied</span>
          <span class="bl-item"><span style="font-family:var(--mono);font-size:9px;color:#34d399;background:rgba(52,211,153,0.1);padding:1px 5px;border-radius:3px">×2.4</span> Symptom <strong>supports</strong> this finding &nbsp;·&nbsp; <span style="font-family:var(--mono);font-size:9px;color:#f87171;background:rgba(248,113,113,0.1);padding:1px 5px;border-radius:3px">×0.4</span> Symptom argues <strong>against</strong> it</span>
          <span class="bl-item"><span style="font-family:var(--mono);font-size:9px;color:#94a3b8">±x%</span> = AI uncertainty (MC Dropout, 20 passes) — smaller is more confident</span>
        </div>
        <div id="prob-bars"></div>
      </div>
      <div id="radar-view">
        <div class="radar-wrap">
          <canvas id="radarCanvas" width="500" height="400"></canvas>
        </div>
        <div class="radar-legend" id="radar-legend"></div>
      </div>
    </div>

    <!-- FILM SYMPTOM PROOF PANEL -->
    <div id="film-proof-box" style="display:none"></div>

    <!-- CLINICAL REPORTS -->
    <div id="clinical-report"></div>

    <div class="disclaimer">
      ⚠️ <span data-en>AyShCXR is a research prototype for clinical assistance only. All outputs must be reviewed by a qualified medical professional. NOT for standalone diagnosis.</span>
      <span data-hi style="display:none">AyShCXR केवल नैदानिक सहायता के लिए एक शोध प्रोटोटाइप है। सभी परिणामों की समीक्षा एक योग्य चिकित्सा पेशेवर द्वारा की जानी चाहिए।</span>
    </div>

    <div class="action-row">
      <button class="act-btn export" onclick="exportReport()">📄 <span data-en>Export Report</span><span data-hi style="display:none">रिपोर्ट निर्यात</span></button>
      <button class="act-btn pdf" onclick="printReport()">🖨️ <span data-en>Print / Share</span><span data-hi style="display:none">प्रिंट / शेयर</span></button>
      <button class="act-btn clear" onclick="clearAll()">← <span data-en>New Analysis</span><span data-hi style="display:none">नया विश्लेषण</span></button>
    </div>
  </div>

</main>
</div>

<script>
// ═══ LANGUAGE TOGGLE ═══
let currentLang='en';
function toggleLang(){
  currentLang=currentLang==='en'?'hi':'en';
  const root=document.getElementById('app-root');
  if(currentLang==='hi'){
    root.classList.add('hindi-mode');
    document.getElementById('btn-en').classList.remove('active');
    document.getElementById('btn-hi').classList.add('active');
  }else{
    root.classList.remove('hindi-mode');
    document.getElementById('btn-en').classList.add('active');
    document.getElementById('btn-hi').classList.remove('active');
  }
}

// ═══ DEMO MODE ═══
function showDemo(){document.getElementById('demo-panel').style.display='block';document.getElementById('demo-panel').scrollIntoView({behavior:'smooth'});}
function hideDemo(){document.getElementById('demo-panel').style.display='none';}
function loadDemo(disease){
  alert(`Demo mode: In a full deployment, this would load a pre-verified NIH test X-ray for ${disease} and run the complete two-stage analysis automatically.\n\nFor now please upload a real NIH chest X-ray from your test set.`);
  hideDemo();
}

// ═══ SYMPTOM BUILD ═══
const STAGE1_CATEGORIES=[
  {category:"Breathlessness",categoryHi:"सांस फूलना",icon:"🫁",symptoms:[
    {id:"breathless_mild",    label:"Mild",                     labelHi:"हल्की सांस फूलना"},
    {id:"breathless_moderate",label:"Moderate",                 labelHi:"मध्यम सांस फूलना",sub:true},
    {id:"breathless_severe",  label:"Severe",                   labelHi:"गंभीर सांस फूलना",sub:true},
    {type:"dur",group:"breathless_dur",label:"Duration",labelHi:"अवधि",
     options:[
       {id:"breathless_dur_acute",   label:"< 3 days",  labelHi:"< 3 दिन"},
       {id:"breathless_dur_subacute",label:"3 days – 2 wk",labelHi:"3 दिन – 2 सप्ताह"},
       {id:"breathless_dur_chronic", label:"> 2 weeks", labelHi:"> 2 सप्ताह"}
     ]}
  ]},
  {category:"Cough & Sputum",categoryHi:"खांसी और बलगम",icon:"🤧",symptoms:[
    {id:"cough_dry",          label:"Dry Cough",                labelHi:"सूखी खांसी"},
    {id:"cough_productive",   label:"Productive Cough",         labelHi:"बलगम वाली खांसी"},
    {id:"cough_bloodstreaked",label:"Blood-streaked Sputum",    labelHi:"खून मिला बलगम",sub:true},
    {id:"sputum_purulent",    label:"Purulent Sputum",          labelHi:"पीप वाला बलगम",sub:true},
    {id:"haemoptysis_present",label:"Coughing Blood",           labelHi:"खून की खांसी"},
    {id:"haemoptysis_frank",  label:"Frank Haemoptysis",        labelHi:"भारी खून खांसी",sub:true},
    {id:"wheezing",           label:"Wheezing",                 labelHi:"घरघराहट"},
    {type:"dur",group:"cough_dur",label:"Cough Duration",labelHi:"खांसी की अवधि",
     options:[
       {id:"cough_dur_short", label:"< 1 week",   labelHi:"< 1 सप्ताह"},
       {id:"cough_dur_medium",label:"1 – 3 weeks", labelHi:"1 – 3 सप्ताह"},
       {id:"cough_dur_long",  label:"> 3 weeks",  labelHi:"> 3 सप्ताह"}
     ]}
  ]},
  {category:"Chest",categoryHi:"छाती",icon:"💗",symptoms:[
    {id:"chest_pain",         label:"Chest Pain",               labelHi:"छाती दर्द"},
    {id:"chest_tightness",    label:"Chest Tightness",          labelHi:"छाती में जकड़न"},
    {id:"pleuritic_pain",     label:"Pain on Breathing",        labelHi:"सांस में दर्द"},
    {id:"palpitations",       label:"Heart Palpitations",       labelHi:"दिल की धड़कन"},
    {id:"swelling",           label:"Leg / Ankle Swelling",     labelHi:"पैर सूजन"}
  ]},
  {category:"General",categoryHi:"सामान्य",icon:"🌡️",symptoms:[
    {id:"fever_low",          label:"Low-grade Fever",          labelHi:"हल्का बुखार"},
    {id:"fever_high",         label:"High Fever",               labelHi:"तेज बुखार",sub:true},
    {type:"dur",group:"fever_dur",label:"Fever Duration",labelHi:"बुखार की अवधि",
     options:[
       {id:"fever_dur_short", label:"< 1 week",   labelHi:"< 1 सप्ताह"},
       {id:"fever_dur_medium",label:"1 – 3 weeks", labelHi:"1 – 3 सप्ताह"},
       {id:"fever_dur_long",  label:"> 3 weeks",  labelHi:"> 3 सप्ताह"}
     ]},
    {id:"fatigue_present",    label:"Fatigue",                  labelHi:"थकान"},
    {id:"fatigue_severe",     label:"Severe Fatigue",           labelHi:"गंभीर थकान",sub:true},
    {id:"weight_loss",        label:"Weight Loss",              labelHi:"वजन घटना"},
    {id:"night_sweats",       label:"Night Sweats",             labelHi:"रात को पसीना"},
    {id:"loss_appetite",      label:"Loss of Appetite",         labelHi:"भूख न लगना"},
    {id:"confusion",          label:"Confusion",                labelHi:"भ्रम"}
  ]},
  {category:"Risk Factors",categoryHi:"जोखिम कारक",icon:"⚠️",symptoms:[
    {id:"tb_contact",         label:"TB Contact",               labelHi:"टीबी संपर्क"},
    {id:"recent_surgery",     label:"Recent Surgery",           labelHi:"हाल की सर्जरी"}
  ]}
];

let _sympTotal=0;
function _updateSympCount(){
  let n=0;
  document.querySelectorAll("#symptoms-container input[type=checkbox]:checked").forEach(()=>n++);
  const el=document.getElementById("symp-total-count");
  if(el){el.textContent=n>0?`${n} symptom${n>1?"s":""} selected`:"No symptoms selected";}
}
const container=document.getElementById("symptoms-container");
STAGE1_CATEGORIES.forEach(cat=>{
  const catLabel=document.createElement("div");
  catLabel.className="symptom-cat";
  catLabel.innerHTML=`<span class="symptom-cat-icon">${cat.icon||"•"}</span><span class="symptom-cat-text"><span data-en>${cat.category}</span><span data-hi style="display:none">${cat.categoryHi}</span></span><span class="symptom-cat-line"></span>`;
  container.appendChild(catLabel);
  const grid=document.createElement("div");
  grid.className="symptoms-grid";
  cat.symptoms.forEach(s=>{
    if(s.type==="dur"){
      // Duration radio pill row — mutually exclusive, full-width
      const row=document.createElement("div");
      row.className="dur-row";
      row.dataset.group=s.group;
      const rowLbl=document.createElement("div");
      rowLbl.className="dur-row-label";
      rowLbl.innerHTML=`<span data-en>${s.label}</span><span data-hi style="display:none">${s.labelHi}</span>`;
      row.appendChild(rowLbl);
      const pills=document.createElement("div");
      pills.className="dur-pills";
      s.options.forEach(opt=>{
        const pill=document.createElement("label");
        pill.className="dur-pill";
        pill.innerHTML=`<input type="radio" name="dur-${s.group}" id="dur-${opt.id}" value="${opt.id}">
          <span data-en>${opt.label}</span><span data-hi style="display:none">${opt.labelHi}</span>`;
        pill.querySelector("input").addEventListener("change",()=>{
          pills.querySelectorAll(".dur-pill").forEach(p=>p.classList.remove("selected"));
          pill.classList.add("selected");
          _updateSympCount();
        });
        pills.appendChild(pill);
      });
      row.appendChild(pills);
      grid.appendChild(row);
      return;
    }
    const lbl=document.createElement("label");
    lbl.className="symptom-item"+(s.sub?" sub-item":"");
    lbl.innerHTML=`<input type="checkbox" id="sym-${s.id}" value="${s.id}">
      <span data-en>${s.label}</span><span data-hi style="display:none">${s.labelHi}</span>`;
    lbl.querySelector("input").addEventListener("change",e=>{
      lbl.classList.toggle("checked",e.target.checked);
      _updateSympCount();
    });
    grid.appendChild(lbl);
  });
  container.appendChild(grid);
});
const _sympFooter=document.createElement("div");
_sympFooter.className="symp-total";
_sympFooter.innerHTML=`<span id="symp-total-count">No symptoms selected</span>`;
container.appendChild(_sympFooter);

// ═══ FILE HANDLING ═══
let selectedFile=null,stage1Data=null,stage2Qs=[],stage2Answers={},reportData=null;
let isCameraCapture=false;

const _DUR_LABELS=["Not specified","< 1 week","1–2 weeks","2–4 weeks","1–3 months","> 3 months"];
const _DUR_VALS  =[null,           "<1wk",     "1-2wk",    "2-4wk",    "1-3mo",     ">3mo"];
function updateDuration(v){
  const idx=parseInt(v)+1;
  const lbl=_DUR_LABELS[idx];
  document.getElementById("dur-badge").textContent=lbl;
  // Sync to patient duration field so it's included in report
  const ptDur=document.getElementById("pt-duration");
  if(ptDur&&v>0) ptDur.value=lbl;
  // Animate track fill
  const pct=(parseInt(v)/4)*100;
  document.getElementById("sym-dur-range").style.background=
    `linear-gradient(90deg,var(--accent) ${pct}%,rgba(255,255,255,0.08) ${pct}%)`;
}

function handleFile(input){
  if(!input.files[0])return;
  isCameraCapture=false;
  selectedFile=input.files[0];
  processFile(selectedFile);
}

function handleCameraFile(input){
  if(!input.files[0])return;
  isCameraCapture=true;
  selectedFile=input.files[0];
  document.getElementById('preprocess-status').style.display='block';
  processFile(selectedFile);
}

function processFile(file){
  document.getElementById("validation-error").style.display="none";
  document.getElementById("upload-zone").classList.remove("invalid");
  const reader=new FileReader();
  reader.onload=e=>{
    const img=document.getElementById("preview-img");
    img.src=e.target.result;
    img.style.display="block";
    document.getElementById("upload-icon").textContent="✅";
    document.getElementById("upload-h3").textContent=file.name;
    document.getElementById("upload-p").textContent="Ready to analyse";
  };
  reader.readAsDataURL(file);
}

function handleDrag(e){e.preventDefault();document.getElementById("upload-zone").classList.add("dragging");}
function handleDragLeave(){document.getElementById("upload-zone").classList.remove("dragging");}
function handleDrop(e){
  e.preventDefault();
  document.getElementById("upload-zone").classList.remove("dragging");
  if(e.dataTransfer.files[0]){
    document.getElementById("file-input").files=e.dataTransfer.files;
    handleFile(document.getElementById("file-input"));
  }
}

function getSymptoms(){
  const checked={};
  STAGE1_CATEGORIES.forEach(cat=>cat.symptoms.forEach(s=>{
    if(s.type==="dur"){
      // One radio may be selected — set its id to true, others false
      s.options.forEach(opt=>{
        const el=document.getElementById(`dur-${opt.id}`);
        checked[opt.id]=el?el.checked:false;
      });
      return;
    }
    const el=document.getElementById(`sym-${s.id}`);
    checked[s.id]=el?el.checked:false;
  }));
  return checked;
}

// ═══ LOADING — STEP PROGRESS ═══
let stepTimer=null,currentStep=0;
const STEP_TIMINGS=[0,800,2000,3200,4500];

function startLoading(msgs){
  const steps=document.querySelectorAll('.step-item');
  steps.forEach(s=>{s.classList.remove('active','done');});
  currentStep=0;
  document.getElementById('loading-msg').textContent='';
  document.getElementById("loading").style.display="block";

  function activateStep(i){
    if(i>=steps.length)return;
    if(i>0)steps[i-1].classList.add('done');
    steps[i].classList.add('active');
    document.getElementById('loading-msg').textContent=steps[i].querySelector('.step-label').textContent;
    stepTimer=setTimeout(()=>activateStep(i+1),1200);
  }
  activateStep(0);
}

function stopLoading(){
  if(stepTimer)clearTimeout(stepTimer);
  const steps=document.querySelectorAll('.step-item');
  steps.forEach(s=>s.classList.add('done'));
  setTimeout(()=>{
    document.getElementById("loading").style.display="none";
    steps.forEach(s=>s.classList.remove('active','done'));
  },400);
}

// ═══ STAGE INDICATOR ═══
function setStage(n){
  for(let i=1;i<=3;i++){
    const c=document.getElementById(`pill-${i}`);
    const l=document.getElementById(`lbl-${i}`);
    if(i<n){c.className="stage-circle complete";l.className="stage-label complete";}
    else if(i===n){c.className="stage-circle active";l.className="stage-label active";}
    else{c.className="stage-circle";l.className="stage-label";}
  }
}

// ═══ STAGE 1 ═══
async function runStage1(){
  if(!selectedFile){alert("Please upload a chest X-ray image first.");return;}
  document.getElementById("analyse-btn").disabled=true;
  document.getElementById("stage2-section").style.display="none";
  document.getElementById("results").style.display="none";
  document.getElementById("validation-error").style.display="none";
  startLoading();
  const formData=new FormData();
  formData.append("image",selectedFile);
  formData.append("name",document.getElementById("pt-name").value||"Anonymous");
  formData.append("age",document.getElementById("pt-age").value||"Unknown");
  formData.append("gender",document.getElementById("pt-gender").value);
  formData.append("duration",document.getElementById("pt-duration").value||"Not specified");
  formData.append("smoking",document.getElementById("pt-smoking").value);
  formData.append("occupation",document.getElementById("pt-occupation").value||"");
  formData.append("conditions",document.getElementById("pt-conditions").value||"");
  formData.append("medications",document.getElementById("pt-medications").value||"");
  formData.append("doctor",document.getElementById("pt-doctor").value||"");
  formData.append("notes",document.getElementById("pt-notes").value||"");
  formData.append("patient_id",document.getElementById("pt-id").value||"");
  formData.append("symptoms",JSON.stringify(getSymptoms()));
  formData.append("camera_capture",isCameraCapture?"1":"0");
  try{
    const resp=await fetch("/predict_stage1",{method:"POST",body:formData});
    const data=await resp.json();
    stopLoading();
    document.getElementById("analyse-btn").disabled=false;
    document.getElementById('preprocess-status').style.display='none';
    if(data.validation_failed){
      document.getElementById("validation-error").textContent="⚠ "+data.error;
      document.getElementById("validation-error").style.display="block";
      document.getElementById("upload-zone").classList.add("invalid");
      return;
    }
    if(data.error){alert("Error: "+data.error);return;}
    stage1Data=data;
    if(data.model_name)document.getElementById("model-badge").textContent=data.model_name;
    setStage(2);
    buildStage2UI(data);
    document.getElementById("stage2-section").style.display="block";
    document.getElementById("stage2-section").scrollIntoView({behavior:"smooth",block:"start"});
  }catch(err){
    stopLoading();
    document.getElementById("analyse-btn").disabled=false;
    alert("Error: "+err.message);
  }
}

// ═══ STAGE 2 ═══
// ═══ FILM PROOF PANEL ═══
function buildFilmProof(predictions){
  const box=document.getElementById("film-proof-box");
  if(!box)return;

  // Only show rows with meaningful FiLM delta (>=1%)
  const rows=predictions
    .filter(p=>p.pre_film!=null && Math.abs(p.probability - p.pre_film)>=0.01)
    .sort((a,b)=>Math.abs(b.probability-b.pre_film)-Math.abs(a.probability-a.pre_film))
    .slice(0,7);

  if(rows.length===0){box.style.display="none";return;}

  const meanDelta=rows.reduce((s,p)=>s+(p.probability-p.pre_film),0)/rows.length;
  const sign=meanDelta>=0?"+":"";

  let rowsHTML=rows.map(p=>{
    const pre=(p.pre_film*100).toFixed(1);
    const post=(p.probability*100).toFixed(1);
    const delta=p.probability-p.pre_film;
    const ds=(delta>=0?"+":"")+( delta*100).toFixed(1)+"%";
    const dcls=delta>=0?"pos":"neg";

    // Audit trail: which symptom moved this, and by what likelihood ratio.
    // LR>1 argues FOR the finding, LR<1 argues AGAINST it. This is what lets a
    // clinician check the reasoning instead of trusting a bare number.
    let ev="";
    if(p.fusion && p.fusion.length){
      ev=`<div class="fpr-evidence">`+p.fusion.map(e=>{
        const up=e.lr>1;
        const col=up?"#34d399":"#f87171";
        const bg=up?"rgba(52,211,153,0.10)":"rgba(248,113,113,0.10)";
        const nm=e.symptom.replace(/_/g," ");
        const ab=e.present?"":" <em>absent</em>";
        return `<span style="font-family:var(--mono);font-size:9px;color:${col};background:${bg};padding:2px 6px;border-radius:3px;margin:2px 3px 0 0;display:inline-block">${nm}${ab} ×${e.lr.toFixed(2)}</span>`;
      }).join("")+`</div>`;
    }

    return `
    <div class="fpr-row">
      <span class="fpr-disease">${p.disease}</span>
      <div class="fpr-bars">
        <div class="fpr-bar-row">
          <span class="fpr-label">Image only</span>
          <div class="fpr-track"><div class="fpr-fill-pre" style="width:${Math.min(parseFloat(pre),100)}%"></div></div>
          <span class="fpr-pct">${pre}%</span>
        </div>
        <div class="fpr-bar-row">
          <span class="fpr-label">With symptoms</span>
          <div class="fpr-track"><div class="fpr-fill-post" style="width:${Math.min(parseFloat(post),100)}%"></div></div>
          <span class="fpr-pct">${post}%</span>
        </div>
        ${ev}
      </div>
      <span class="fpr-delta ${dcls}">${ds}</span>
    </div>`;
  }).join("");

  box.style.display="block";
  box.innerHTML=`
  <div class="film-proof">
    <div class="film-proof-hdr">
      <span class="film-proof-title">⚗ Symptom Fusion — Bayesian Likelihood Ratios</span>
      <span class="film-proof-auc-badge ${meanDelta>=0?'pos-badge':'neg-badge'}">Net shift: ${sign}${(meanDelta*100).toFixed(1)}% on this patient</span>
    </div>

    <div class="film-how-it-works">
      <div class="film-hiw-step">
        <span class="film-hiw-num">1</span>
        <div><strong>Image Only (grey bar)</strong><br>DenseNet-121 reads the X-ray pixels alone — no clinical context.</div>
      </div>
      <div class="film-hiw-arrow">→</div>
      <div class="film-hiw-step">
        <span class="film-hiw-num">2</span>
        <div><strong>Likelihood Ratios</strong><br>Each reported symptom multiplies the odds by its published LR — how much commoner it is with this disease than without. Strong evidence swings hard, weak evidence barely moves it.</div>
      </div>
      <div class="film-hiw-arrow">→</div>
      <div class="film-hiw-step">
        <span class="film-hiw-num">3</span>
        <div><strong>With Symptoms (coloured bar)</strong><br>Adjusted probability. <span style="color:#34d399">Green = symptoms confirm finding.</span> <span style="color:#f87171">Red = symptoms rule it out.</span></div>
      </div>
    </div>

    <div class="film-proof-sub">
      Every likelihood ratio is derived from published clinical prevalence data
      (Harrison's Internal Medicine and related guidelines) — nothing here is
      trained or fitted, so the reasoning is fully auditable.
      <strong style="color:var(--success)">This layer refines and explains the decision; it does not alter the model's reported AUC of 0.8031</strong>,
      which measures image analysis alone.
    </div>

    ${rowsHTML}
    <div class="film-proof-divider"></div>
    <div class="film-proof-footer">
      <span>
        <span class="film-legend-dot grey-dot"></span> Grey = DenseNet-121 image alone &nbsp;·&nbsp;<!--LEGEND-->

        <span class="film-legend-dot teal-dot"></span> Coloured = after reported symptoms applied as likelihood ratios
      </span>
      <span style="color:var(--muted);font-size:9px">Bayesian likelihood-ratio fusion · no trained weights</span>
    </div>
  </div>`;
}

function buildStage2UI(data){
  stage2Qs=data.stage2_questions||[];
  stage2Answers={};
  const topDiv=document.getElementById("stage2-top-diseases");
  topDiv.innerHTML="";
  (data.top_diseases||[]).forEach(d=>{
    const chip=document.createElement("div");
    chip.className="dchip";
    const prob=data.predictions.find(p=>p.disease===d);
    chip.textContent=d+(prob?` — ${(prob.probability*100).toFixed(0)}%`:"");
    topDiv.appendChild(chip);
  });
  const qc=document.getElementById("stage2-questions-container");
  qc.innerHTML="";
  stage2Qs.forEach((q,idx)=>{
    const qDiv=document.createElement("div");
    qDiv.className="s2-q";
    qDiv.innerHTML=`<div class="s2-q-tag">For: ${q.for_disease}</div>
      <div class="s2-q-text">Q${idx+1}. ${q.text}</div>
      <div class="yn-row">
        <button class="yn-btn yes" onclick="answerQuestion('${q.id}',true,this)">✓ Yes</button>
        <button class="yn-btn no"  onclick="answerQuestion('${q.id}',false,this)">✗ No</button>
      </div>`;
    qc.appendChild(qDiv);
  });
  updateS2Progress();
}

function answerQuestion(qId,answer,btn){
  stage2Answers[qId]=answer;
  btn.closest(".yn-row").querySelectorAll(".yn-btn").forEach(b=>b.classList.remove("selected"));
  btn.classList.add("selected");
  updateS2Progress();
}

function updateS2Progress(){
  const total=stage2Qs.length,answered=Object.keys(stage2Answers).length;
  const pct=total>0?(answered/total*100):0;
  document.getElementById("stage2-progress-text").textContent=`${answered} of ${total} questions answered`;
  document.getElementById("stage2-progress-fill").style.width=pct+"%";
  document.getElementById("stage2-submit-btn").disabled=(answered<total);
}

async function runStage2(){
  document.getElementById("stage2-submit-btn").disabled=true;
  startLoading(["Applying clinical verification...","Scoring evidence...","Finding primary diagnosis...","Compiling report..."]);
  try{
    const resp=await fetch("/predict_stage2",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({stage1_data:stage1Data,stage2_answers:stage2Answers})});
    const data=await resp.json();
    stopLoading();
    if(data.error){alert("Error: "+data.error);return;}
    reportData=data;
    setStage(3);
    document.getElementById("stage2-section").style.display="none";
    document.getElementById("results").style.display="block";
    renderResults(data);
    document.getElementById("results").scrollIntoView({behavior:"smooth"});
  }catch(err){
    stopLoading();
    document.getElementById("stage2-submit-btn").disabled=false;
    alert("Error: "+err.message);
  }
}

// ═══ TRAFFIC LIGHT ═══
function renderTrafficLight(primaryPred,advice){
  const box=document.getElementById("traffic-light-box");
  if(!primaryPred){box.innerHTML="";return;}

  const score=primaryPred.stage2_score||primaryPred.probability;
  const info=advice&&advice.find(r=>r.disease===primaryPred.disease)||{};
  const urgency=info.urgency||"low";

  let color,actionEn,actionHi,detailEn,detailHi;

  if(urgency==="high"||score>=0.70){
    color="red";
    actionEn="REFER TO DISTRICT HOSPITAL — URGENTLY TODAY";
    actionHi="जिला अस्पताल में तुरंत रेफर करें — आज ही";
    detailEn=`${info.full_name||primaryPred.disease} detected with high confidence. Do not delay. Contact nearest district hospital immediately.`;
    detailHi=`${info.full_name||primaryPred.disease} उच्च विश्वास के साथ पाया गया। देरी न करें। निकटतम जिला अस्पताल से तुरंत संपर्क करें।`;
  }else if(urgency==="medium"||score>=0.50){
    color="amber";
    actionEn="MONITOR — Follow Up Within 2 Weeks";
    actionHi="निगरानी करें — 2 सप्ताह के भीतर जांच";
    detailEn=`${info.full_name||primaryPred.disease} suspected. Monitor symptoms closely. Schedule follow-up appointment within 2 weeks.`;
    detailHi=`${info.full_name||primaryPred.disease} संदिग्ध। लक्षणों पर नज़र रखें। 2 सप्ताह के भीतर जांच करें।`;
  }else{
    color="green";
    actionEn="LOW RISK — Routine Care";
    actionHi="कम जोखिम — नियमित देखभाल";
    detailEn="No significant findings detected. Continue routine care and advise patient to return if symptoms worsen.";
    detailHi="कोई महत्वपूर्ण निष्कर्ष नहीं पाया गया। नियमित देखभाल जारी रखें।";
  }

  box.innerHTML=`
    <div class="traffic-light-card tl-${color}">
      <div class="tl-lights">
        <div class="tl-light red-bulb${color==='red'?' on':''}"></div>
        <div class="tl-light amber-bulb${color==='amber'?' on':''}"></div>
        <div class="tl-light green-bulb${color==='green'?' on':''}"></div>
      </div>
      <div class="tl-action ${color}" data-en>${actionEn}</div>
      <div class="tl-action ${color}" data-hi style="display:none">${actionHi}</div>
      <div class="tl-detail" data-en>${detailEn}</div>
      <div class="tl-detail" data-hi style="display:none">${detailHi}</div>
      <div class="tl-disease ${color}">${info.full_name||primaryPred.disease} · ${(score*100).toFixed(1)}% confidence</div>
    </div>`;

  // Apply hindi mode if active
  if(currentLang==='hi'){
    box.querySelectorAll('[data-en]').forEach(el=>el.style.display='none');
    box.querySelectorAll('[data-hi]').forEach(el=>el.style.display='block');
  }
}

// ═══ CONFIDENCE GAUGE ═══
function drawGauge(pct,color){
  const canvas=document.getElementById('gaugeCanvas');
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,180,100);
  const cx=90,cy=90,r=70;
  const startAngle=Math.PI,endAngle=2*Math.PI;

  // Background arc
  ctx.beginPath();
  ctx.arc(cx,cy,r,startAngle,endAngle);
  ctx.strokeStyle='rgba(255,255,255,0.06)';
  ctx.lineWidth=12;
  ctx.lineCap='round';
  ctx.stroke();

  // Colored arc
  const valueAngle=startAngle+(endAngle-startAngle)*(pct/100);
  ctx.beginPath();
  ctx.arc(cx,cy,r,startAngle,valueAngle);
  ctx.strokeStyle=color;
  ctx.lineWidth=12;
  ctx.lineCap='round';
  ctx.stroke();

  // Glow
  ctx.beginPath();
  ctx.arc(cx,cy,r,startAngle,valueAngle);
  ctx.strokeStyle=color+'44';
  ctx.lineWidth=20;
  ctx.stroke();

  // Needle dot
  const needleX=cx+r*Math.cos(valueAngle);
  const needleY=cy+r*Math.sin(valueAngle);
  ctx.beginPath();
  ctx.arc(needleX,needleY,6,0,2*Math.PI);
  ctx.fillStyle=color;
  ctx.fill();
  ctx.shadowColor=color;
  ctx.shadowBlur=15;
  ctx.fill();
}

function renderConfidence(primaryPred){
  if(!primaryPred)return;
  const score=(primaryPred.stage2_score||primaryPred.probability)*100;
  const unc=(primaryPred.uncertainty||0)*100;
  const netConf=Math.max(0,score-unc*1.5);

  let color,title,desc,warn;
  if(netConf>=70){
    color='#2ed573';title='High Confidence';
    desc='The AI is confident in this diagnosis. Clinical findings are consistent.';
  }else if(netConf>=45){
    color='#ffa502';title='Moderate Confidence';
    desc='The AI shows moderate confidence. Consider additional clinical correlation.';
    warn='Radiologist review recommended for confirmation.';
  }else{
    color='#ff4757';title='Low Confidence';
    desc='The AI is uncertain. Multiple diagnoses are possible.';
    warn='Radiologist review is strongly recommended before acting on this result.';
  }

  document.getElementById('gauge-pct').textContent=Math.round(netConf)+'%';
  document.getElementById('gauge-pct').style.color=color;
  document.getElementById('conf-title').textContent=title;
  document.getElementById('conf-title').style.color=color;
  document.getElementById('conf-desc').textContent=desc;
  const warnEl=document.getElementById('conf-warn');
  if(warn){warnEl.textContent='⚠ '+warn;warnEl.style.display='block';}
  else{warnEl.style.display='none';}

  drawGauge(netConf,color);
}

// ═══ RADAR CHART ═══
let radarData=null;
function switchViz(type,tab){
  document.querySelectorAll('.viz-tab').forEach(t=>t.classList.remove('active'));
  tab.classList.add('active');
  if(type==='radar'){
    document.getElementById('bars-view').style.display='none';
    document.getElementById('radar-view').style.display='flex';
    if(radarData)drawRadar(radarData);
  }else{
    document.getElementById('bars-view').style.display='block';
    document.getElementById('radar-view').style.display='none';
  }
}

let _barsExpanded=false;
function toggleAllBars(){
  const hidden=document.getElementById('bars-hidden');
  const btn=document.getElementById('bars-toggle-btn');
  if(!hidden||!btn) return;
  _barsExpanded=!_barsExpanded;
  if(_barsExpanded){
    hidden.style.display='block';
    btn.innerHTML='▲ SHOW LESS';
    btn.style.color='var(--accent)';
    btn.style.borderColor='var(--accent)';
  } else {
    hidden.style.display='none';
    btn.innerHTML=btn.getAttribute('data-label')||btn.innerHTML.replace('▲ SHOW LESS','▼ SHOW ALL');
    btn.style.color='var(--muted)';
    btn.style.borderColor='var(--glass-border)';
  }
}

function drawRadar(predictions){
  const canvas=document.getElementById('radarCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const cx=W/2,cy=H/2,maxR=Math.min(cx,cy)-60;
  const n=predictions.length;
  const angles=predictions.map((_,i)=>i*(2*Math.PI/n)-Math.PI/2);

  // Grid circles
  [0.25,0.5,0.75,1.0].forEach(r=>{
    ctx.beginPath();
    for(let i=0;i<n;i++){
      const x=cx+maxR*r*Math.cos(angles[i]);
      const y=cy+maxR*r*Math.sin(angles[i]);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.strokeStyle=`rgba(255,255,255,${r===1?0.12:0.05})`;
    ctx.lineWidth=1;
    ctx.stroke();
    // Label
    ctx.fillStyle='rgba(100,116,139,0.6)';
    ctx.font='9px JetBrains Mono';
    ctx.fillText((r*100).toFixed(0)+'%',cx+4,cy-maxR*r+4);
  });

  // Spokes
  angles.forEach((angle,i)=>{
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.lineTo(cx+maxR*Math.cos(angle),cy+maxR*Math.sin(angle));
    ctx.strokeStyle='rgba(255,255,255,0.05)';
    ctx.stroke();
    // Disease labels
    const labelR=maxR+28;
    const lx=cx+labelR*Math.cos(angle);
    const ly=cy+labelR*Math.sin(angle);
    ctx.fillStyle=predictions[i].is_primary?'#00d4ff':'rgba(100,116,139,0.8)';
    ctx.font=predictions[i].is_primary?'bold 10px JetBrains Mono':'9px JetBrains Mono';
    ctx.textAlign='center';
    ctx.textBaseline='middle';
    const name=predictions[i].disease.replace('Pleural Thickening','Pl.Thick');
    ctx.fillText(name,lx,ly);
  });

  // Data polygon
  const scores=predictions.map(p=>p.stage2_score||p.probability);
  ctx.beginPath();
  scores.forEach((s,i)=>{
    const x=cx+maxR*s*Math.cos(angles[i]);
    const y=cy+maxR*s*Math.sin(angles[i]);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.closePath();
  ctx.fillStyle='rgba(0,212,255,0.12)';
  ctx.fill();
  ctx.strokeStyle='rgba(0,212,255,0.6)';
  ctx.lineWidth=2;
  ctx.stroke();

  // Dots
  scores.forEach((s,i)=>{
    const x=cx+maxR*s*Math.cos(angles[i]);
    const y=cy+maxR*s*Math.sin(angles[i]);
    ctx.beginPath();
    ctx.arc(x,y,predictions[i].is_primary?6:4,0,2*Math.PI);
    ctx.fillStyle=predictions[i].is_primary?'#00d4ff':
                  s>=0.5?'#ff4757':s>=0.35?'#ffa502':'rgba(100,116,139,0.5)';
    ctx.fill();
    if(predictions[i].is_primary){
      ctx.shadowColor='#00d4ff';ctx.shadowBlur=12;ctx.fill();ctx.shadowBlur=0;
    }
  });
}

// ═══ RENDER RESULTS ═══
function renderResults(data){
  const reader=new FileReader();
  reader.onload=e=>{document.getElementById("result-xray").src=e.target.result;};
  reader.readAsDataURL(selectedFile);
  if(data.heatmap)document.getElementById("result-heatmap").src="data:image/png;base64,"+data.heatmap;

  const primaryPred=data.predictions.find(p=>p.is_primary);

  // Update heatmap label with disease name
  if(primaryPred){
    document.getElementById('heatmap-label').textContent=`AI Focus — ${primaryPred.disease}`;
  }

  // Traffic light — FIRST thing rendered
  renderTrafficLight(primaryPred,data.advice);

  // Confidence meter
  renderConfidence(primaryPred);

  // Primary finding card
  const primaryBox=document.getElementById("primary-finding-box");
  if(primaryPred){
    const info=data.advice&&data.advice.find(r=>r.disease===primaryPred.disease)||{};
    const imgScore=(primaryPred.probability*100).toFixed(1);
    const finalScore=(primaryPred.stage2_score*100).toFixed(1);
    const delta=(primaryPred.stage2_delta*100).toFixed(1);
    primaryBox.style.display="block";
    primaryBox.innerHTML=`
      <div class="primary-card">
        <div class="primary-tag">🎯 Primary Finding — Stage 2 Confirmed</div>
        <div class="primary-name">${info.full_name||primaryPred.disease}</div>
        <div class="primary-icd">ICD-10: ${info.icd10||"—"}</div>
        <div class="score-row">
          <div class="score-block"><div class="score-val c">${imgScore}%</div><div class="score-lbl">AI Image</div></div>
          <div class="score-block"><div class="score-val p">+${delta}%</div><div class="score-lbl">Clinical</div></div>
          <div class="score-block"><div class="score-val g">${finalScore}%</div><div class="score-lbl">Combined</div></div>
        </div>
        <div class="primary-action">→ ${info.urgency_message||"Seek medical evaluation"}</div>
      </div>`;
  }else{primaryBox.style.display="none";}

  // Other findings
  const otherBox=document.getElementById("other-findings-box");
  const others=data.predictions.filter(p=>!p.is_primary&&data.top_diseases&&data.top_diseases.includes(p.disease));
  if(others.length>0){
    otherBox.style.display="block";
    document.getElementById("other-findings-list").innerHTML=others.map(p=>`
      <div class="of-chip">
        <div class="of-name">${p.disease}</div>
        <div class="of-pct">${(p.stage2_score*100).toFixed(1)}% combined</div>
      </div>`).join("");
  }

  // Risk notes
  if(data.risk_notes&&data.risk_notes.length>0){
    document.getElementById("risk-summary-box").style.display="block";
    document.getElementById("risk-summary-list").innerHTML=data.risk_notes.map(n=>`<li>${n}</li>`).join("");
  }

  // Extra diseases
  if(data.extra_diseases&&data.extra_diseases.length>0){
    document.getElementById("extra-diseases-box").style.display="block";
    document.getElementById("extra-diseases-list").innerHTML=data.extra_diseases.map(d=>`<span class="extra-chip">⚕ ${d}</span>`).join("");
  }

  // Probability bars — primary + top findings shown, rest collapsed
  const barsDiv=document.getElementById("prob-bars");
  barsDiv.innerHTML="";
  const thresholds=data.thresholds||{};

  // Sort: primary first, then detected, then by score descending
  const sortedPreds=[...data.predictions].sort((a,b)=>{
    if(a.is_primary) return -1;
    if(b.is_primary) return 1;
    const sa=a.stage2_score||a.probability, sb=b.stage2_score||b.probability;
    const ta=thresholds[a.disease]||{detected:0.50,borderline:0.35};
    const tb=thresholds[b.disease]||{detected:0.50,borderline:0.35};
    const adet=sa>=ta.detected, bdet=sb>=tb.detected;
    if(adet&&!bdet) return -1;
    if(!adet&&bdet) return 1;
    return sb-sa;
  });

  function buildBarHTML(item){
    const score=item.stage2_score||item.probability;
    const pct=(score*100).toFixed(1);
    const unc=item.uncertainty?(item.uncertainty*100).toFixed(1):null;
    const isPrimary=item.is_primary;
    const t=thresholds[item.disease]||{detected:0.50,borderline:0.35};
    const det=!isPrimary&&score>=t.detected;
    const warn=!isPrimary&&score>=t.borderline&&!det;
    const barClass=isPrimary?"bar-primary":det?"bar-detected":warn?"bar-warn":"bar-clear";
    const badgeHtml=isPrimary?`<span class="dbadge dbadge-primary">🎯 PRIMARY</span>`:
                    det?`<span class="dbadge dbadge-danger">⚠ DETECTED</span>`:
                    warn?`<span class="dbadge dbadge-warn">~ BORDERLINE</span>`:"";
    const uncHtml=unc?`<span class="unc-range">±${unc}%</span>`:"";
    const color=isPrimary?"var(--accent)":det?"var(--danger)":warn?"var(--warn)":"var(--muted)";
    // image-only vs after clinical evidence
    let filmHtml="", preFilmBarHtml="";
    if(item.pre_film!=null){
      const delta=score-item.pre_film;
      if(Math.abs(delta)>=0.005){
        const sign=delta>0?"+":"";
        const cls=delta>0?"pos":"neg";
        filmHtml=`<span class="film-delta ${cls}">symptoms ${sign}${(delta*100).toFixed(1)}%</span>`;
        const prePct=(item.pre_film*100).toFixed(1);
        preFilmBarHtml=`<div class="pre-film-track"><div class="pre-film-fill" style="width:${Math.min(parseFloat(prePct),100)}%" title="Image only: ${prePct}%"></div></div>`;
      }
    }
    return `
      <div class="disease-row">
        <div class="disease-hdr">
          <span class="disease-name">${item.disease}${badgeHtml}${filmHtml}</span>
          <span class="disease-right">${uncHtml}<span class="disease-pct" style="color:${color}">${pct}%</span></span>
        </div>
        <div class="bar-track"><div class="bar-fill ${barClass}" style="width:${Math.min(parseFloat(pct),100)}%"></div></div>
        ${preFilmBarHtml}
      </div>`;
  }

  // Show top 3 (primary + 2 next highest) by default
  const SHOW_N=3;
  sortedPreds.slice(0,SHOW_N).forEach(item=>{ barsDiv.innerHTML+=buildBarHTML(item); });

  // Count ruled-out diseases (below borderline threshold)
  const ruledOut=sortedPreds.slice(SHOW_N).filter(i=>{
    const s=i.stage2_score||i.probability;
    return s<(thresholds[i.disease]||{borderline:0.35}).borderline;
  }).length;
  const hiddenCount=sortedPreds.length-SHOW_N;

  // Expand/collapse toggle
  if(hiddenCount>0){
    let hiddenHTML="";
    sortedPreds.slice(SHOW_N).forEach(item=>{ hiddenHTML+=buildBarHTML(item); });
    barsDiv.innerHTML+=`
      <div id="bars-hidden" style="display:none">${hiddenHTML}</div>
      <div style="text-align:center;margin-top:12px">
        <button id="bars-toggle-btn" onclick="toggleAllBars()"
          style="background:none;border:1px solid var(--glass-border);color:var(--muted);
                 font-family:var(--mono);font-size:10px;padding:7px 18px;border-radius:6px;
                 cursor:pointer;letter-spacing:1px;transition:all 0.2s">
          ▼ SHOW ALL ${sortedPreds.length} FINDINGS · ${ruledOut} RULED OUT
        </button>
      </div>`;
  }

  // Store for radar
  radarData=data.predictions;

  // FiLM proof panel
  buildFilmProof(data.predictions);

  // Clinical reports
  const reportDiv=document.getElementById("clinical-report");
  reportDiv.innerHTML="";
  if(data.advice&&data.advice.length>0){
    data.advice.forEach(r=>{
      const isPrimary=r.disease===(primaryPred&&primaryPred.disease);
      const cardClass=isPrimary?"rp":r.urgency==="high"?"rh":r.urgency==="medium"?"rm":"rl";
      const pillClass=isPrimary?"up-primary":r.urgency==="high"?"up-high":r.urgency==="medium"?"up-medium":"up-low";
      const pillLabel=isPrimary?"PRIMARY DIAGNOSIS":r.urgency.toUpperCase()+" URGENCY";
      const treats=r.treatments.slice(0,5).map(t=>`<li>${t}</li>`).join("");
      const labTests=r.lab_tests?r.lab_tests.slice(0,4).map(t=>`<span class="labtag">${t}</span>`).join(""):"";
      const diffDx=r.differential?r.differential.slice(0,3).map(d=>`<span class="rtag">${d}</span>`).join(""):"";
      reportDiv.innerHTML+=`
        <div class="report-card ${cardClass}">
          ${isPrimary?'<div style="font-family:var(--mono);font-size:9px;color:var(--accent);letter-spacing:2px;margin-bottom:10px">🎯 PRIMARY FINDING — STAGE 2 CONFIRMED</div>':''}
          <div class="r-dname">${r.full_name}</div>
          <div class="r-icd">ICD-10: ${r.icd10||"—"}</div>
          <div class="r-prob">AI Probability: ${r.probability}%${r.uncertainty?` ± ${(r.uncertainty*100).toFixed(1)}%`:""}</div>
          <span class="urgency-pill ${pillClass}">${pillLabel}</span>
          <div class="r-section"><h5>Urgency Action</h5><p>→ ${r.urgency_message}</p></div>
          <div class="r-section"><h5>What It Is</h5><p>${r.description}</p></div>
          ${r.imaging_findings?`<div class="r-section"><h5>X-Ray Findings</h5><p>${r.imaging_findings}</p></div>`:""}
          <div class="r-section"><h5>Specialist</h5><div class="specialist-tag">👨‍⚕️ ${r.specialist}</div></div>
          ${diffDx?`<div class="r-section"><h5>Consider Also</h5><div class="tag-row">${diffDx}</div></div>`:""}
          ${labTests?`<div class="r-section"><h5>Investigations</h5><div class="tag-row">${labTests}</div></div>`:""}
          <div class="r-section"><h5>Treatments</h5><ul>${treats}</ul></div>
          <div class="r-section"><h5>Prescription</h5><p>${r.prescription_note}</p></div>
          ${r.follow_up?`<div class="followup-box">📅 <strong>Follow-up:</strong> ${r.follow_up}</div>`:""}
          <div class="emergency-box">🚨 ${r.emergency_signs}</div>
        </div>`;
    });
  }
  if(data.history_advice&&data.history_advice.length>0){
    const sep=document.createElement("div");
    sep.className="separator";
    sep.textContent="Additionally Suspected From Patient History";
    reportDiv.appendChild(sep);
    data.history_advice.forEach(r=>{
      const treats=r.treatments.slice(0,5).map(t=>`<li>${t}</li>`).join("");
      const labTests=r.lab_tests?r.lab_tests.slice(0,4).map(t=>`<span class="labtag">${t}</span>`).join(""):"";
      reportDiv.innerHTML+=`
        <div class="report-card rx">
          <div class="r-dname">${r.full_name}</div>
          <div class="r-icd">ICD-10: ${r.icd10||"—"}</div>
          <span class="history-label">⚕ Suspected from history — not AI detected</span>
          <div class="r-section"><h5>What It Is</h5><p>${r.description}</p></div>
          <div class="r-section"><h5>Investigations</h5><div class="tag-row">${labTests}</div></div>
          <div class="r-section"><h5>Specialist</h5><div class="specialist-tag">👨‍⚕️ ${r.specialist}</div></div>
          <div class="r-section"><h5>Treatments</h5><ul>${treats}</ul></div>
          <div class="emergency-box">🚨 ${r.emergency_signs}</div>
        </div>`;
    });
  }
}

// ═══ EXPORT ═══
function exportReport(){
  if(!reportData)return;
  const p=reportData.patient||{};
  const primaryPred=reportData.predictions.find(pr=>pr.is_primary);
  let text="=".repeat(60)+"\n  AyShCXR — Two-Stage AI Clinical Report\n  by Subhrakant Sethi & Ayush Singh\n  Model: "+(reportData.model_name||"Unknown")+"\n"+"=".repeat(60)+"\n\n";
  text+=`Patient    : ${p.name||"Anonymous"}\nAge/Gender : ${p.age||"—"} / ${p.gender||"—"}\nOccupation : ${p.occupation||"Not specified"}\n\n`;
  if(primaryPred){
    text+="PRIMARY DIAGNOSIS (Stage 2 Confirmed):\n"+"-".repeat(50)+"\n"+primaryPred.disease+
          "\nAI Image Score    : "+(primaryPred.probability*100).toFixed(1)+"%\n"+
          "Clinical Evidence : +"+(primaryPred.stage2_delta*100).toFixed(1)+"%\n"+
          "Combined Score    : "+(primaryPred.stage2_score*100).toFixed(1)+"%\n\n";
  }
  text+="DETECTED FINDINGS:\n"+"-".repeat(50)+"\n";
  const threshR=reportData.thresholds||{};
  reportData.predictions.forEach(p2=>{
    const score=(p2.stage2_score||p2.probability)*100;
    const t=threshR[p2.disease]||{detected:50,borderline:35};
    const isDetected=p2.is_primary||score>=t.detected*100||score>=t.borderline*100;
    if(!isDetected)return;
    const status=p2.is_primary?"PRIMARY":score>=t.detected*100?"DETECTED":"BORDERLINE";
    const filmNote=(p2.pre_film!=null&&Math.abs((p2.stage2_score||p2.probability)-p2.pre_film)>=0.005)?
      `  [FiLM ${((p2.stage2_score||p2.probability)-p2.pre_film)>0?"+":""}${(((p2.stage2_score||p2.probability)-p2.pre_film)*100).toFixed(1)}%]`:"";
    text+=p2.disease.padEnd(22)+score.toFixed(1).padStart(6)+"%  "+status+filmNote+"\n";
  });
  text+="\n"+"=".repeat(60)+"\nDISCLAIMER: Research prototype. NOT for standalone clinical diagnosis.\n"+"=".repeat(60)+"\n";
  const blob=new Blob([text],{type:"text/plain;charset=utf-8"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");a.href=url;a.download=`ayshcxr_report_${Date.now()}.txt`;a.click();URL.revokeObjectURL(url);
}

function printReport(){
  window.print();
}

// ═══ CLEAR ═══
function clearAll(){
  selectedFile=null;stage1Data=null;reportData=null;radarData=null;stage2Qs=[];stage2Answers={};isCameraCapture=false;
  document.getElementById("results").style.display="none";
  document.getElementById("stage2-section").style.display="none";
  document.getElementById("preview-img").style.display="none";
  document.getElementById("file-input").value="";
  document.getElementById("camera-input").value="";
  document.getElementById("validation-error").style.display="none";
  document.getElementById("upload-zone").classList.remove("invalid");
  document.getElementById("upload-icon").textContent="📡";
  document.getElementById("upload-h3").textContent="Upload or Capture Chest X-Ray";
  document.getElementById("upload-p").textContent="Chest radiograph only · PNG JPG · PA or AP view";
  document.getElementById("preprocess-status").style.display="none";
  document.getElementById("traffic-light-box").innerHTML="";
  document.querySelectorAll(".symptom-item").forEach(el=>{el.classList.remove("checked");el.querySelector("input").checked=false;});
  document.querySelectorAll(".dur-pill").forEach(el=>{el.classList.remove("selected");el.querySelector("input").checked=false;});
  ["pt-name","pt-age","pt-duration","pt-notes","pt-occupation","pt-conditions","pt-medications","pt-doctor","pt-id"].forEach(id=>{const el=document.getElementById(id);if(el)el.value="";});
  ["risk-summary-box","extra-diseases-box","primary-finding-box","other-findings-box"].forEach(id=>{document.getElementById(id).style.display="none";});
  setStage(1);
  window.scrollTo({top:0,behavior:"smooth"});
}

// Model info
fetch("/model_info").then(r=>r.json()).then(data=>{
  document.getElementById("model-badge").textContent=data.model_name;
  const pill=document.getElementById("disease-count-pill");
  if(pill&&data.n_diseases) pill.textContent=data.n_diseases+" Findings · v7";
}).catch(()=>{
  document.getElementById("model-badge").textContent="AyShCXR";
  const pill=document.getElementById("disease-count-pill");
  if(pill) pill.textContent="AyShCXR";
});
</script>
</body>
</html>
"""
# ── Routes ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/model_info")
def model_info():
    return jsonify({
        "model_name": model_name,
        "n_diseases": num_diseases,
        "models": [{"id": e["id"], "auc": e["auc"]} for e in LOADED],
        "calibrated": bool(_calib.load()),
    })

@app.route("/predict_stage1", methods=["POST"])
def predict_stage1():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file       = request.files["image"]
    name       = request.form.get("name",        "Anonymous")
    age        = request.form.get("age",         "Unknown")
    gender     = request.form.get("gender",      "Unknown")
    duration   = request.form.get("duration",    "Not specified")
    smoking    = request.form.get("smoking",     "no")
    occupation = request.form.get("occupation",  "")
    conditions = request.form.get("conditions",  "")
    medications= request.form.get("medications", "")
    doctor     = request.form.get("doctor",      "")
    notes      = request.form.get("notes",       "")
    patient_id = request.form.get("patient_id",  "")

    # Malformed JSON here previously raised an uncaught JSONDecodeError, which
    # became an HTTP 500 — and with debug mode on, that served the interactive
    # Werkzeug debugger to the caller. Any client-supplied string must be parsed
    # defensively.
    try:
        symptoms = json.loads(request.form.get("symptoms", "{}") or "{}")
        if not isinstance(symptoms, dict):
            symptoms = {}
    except (ValueError, TypeError):
        print("⚠️  malformed symptoms JSON received — treating as none reported")
        symptoms = {}

    # ── X-ray validation ──────────────────────────────
    try:
        img_bytes = file.stream.read()
        file.stream.seek(0)
        img_pil = Image.open(io.BytesIO(img_bytes))
        # verify() parses headers and rejects a file that merely claims to be an
        # image. It consumes the file object, so reopen afterwards.
        img_pil.verify()
        img_pil = Image.open(io.BytesIO(img_bytes))
        if img_pil.format not in ("PNG", "JPEG", "JPG", "BMP", "TIFF", "WEBP"):
            return jsonify({
                "validation_failed": True,
                "error": f"Unsupported image format ({img_pil.format}). Please "
                         f"upload a PNG or JPEG chest radiograph."
            }), 200
        is_valid, reason = validate_xray(img_pil)
        if not is_valid:
            return jsonify({"validation_failed": True, "error": reason}), 200
        # RGB, not L. The old single model took 1-channel input; the CheXpert
        # models take 3-channel. Each registry transform inserts its own
        # Grayscale step when its model needs one, so RGB is the correct common
        # input for a mixed ensemble. Passing "L" to a 3-channel model fails
        # inside Normalize with a broadcast-shape error.
        img = img_pil.convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {str(e)}"}), 400

    # ── Radiographic technique assessment ─────────────
    # A rotated film widens the heart and fakes cardiomegaly; an over-penetrated
    # film hides nodules. This runs BEFORE diagnosis so the operator is told
    # which findings are unreliable on THIS film. Never fatal — a failure here
    # must not stop the radiograph being read.
    try:
        quality = _iq.assess(img_pil)
    except Exception as e:
        quality = {"available": False, "error": str(e)[:120],
                   "overall": "unknown", "warnings": [], "affected_findings": []}

    # ── AI prediction ─────────────────────────────────
    # The previous fallback here was dangerous: on ANY exception it silently ran
    # the PRIMARY model alone through the old 1-channel transform, producing 14
    # probabilities that were then zipped against 21 canonical disease names —
    # mislabelling every finding, with no error shown to the user.
    #
    # Now: retry once without MC-Dropout (the commonest transient cause is a
    # CUDA OOM from 20 stochastic passes), still through model_loader so label
    # mapping stays correct. If that fails too, return an error rather than
    # invent a result. A clinical tool must refuse rather than guess.
    try:
        mean_probs, std_probs = predict_with_uncertainty(img, n_passes=20)
    except Exception as e_full:
        print(f"⚠️  20-pass inference failed ({type(e_full).__name__}: "
              f"{str(e_full)[:120]}) — retrying single pass")
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            mean_probs, std_probs = predict_with_uncertainty(img, n_passes=0)
        except Exception as e_single:
            print(f"❌ Inference failed: {type(e_single).__name__}: {e_single}")
            return jsonify({
                "error": "The AI could not analyse this image. "
                         f"({type(e_single).__name__}). Please try again, or "
                         "try a different file if the problem persists."
            }), 500

    predictions = [
        {"disease": d, "probability": round(float(mean_probs[i]), 4),
         "uncertainty": round(float(std_probs[i]), 4)}
        for i, d in enumerate(active_diseases)
    ]

    # Risk factors + Bayesian symptom update.
    # `pre_film` keeps its name for JSON/UI compatibility but now means
    # "image-only probability, before any clinical evidence was applied".
    pre_film = {p["disease"]: p["probability"] for p in predictions}
    predictions, detected_risks = apply_symptom_fusion(
        img, predictions, symptoms, smoking, occupation, conditions)
    for p in predictions:
        p["pre_film"] = pre_film.get(p["disease"], p["probability"])

    # GradCAM — must index the PRIMARY model's own outputs, not the canonical list
    top_idx = int(np.argmax(mean_probs))
    top_name = active_diseases[top_idx] if top_idx < len(active_diseases) else None
    heatmap = None
    heatmap_raw = None
    heatmap_disease = None
    try:
        cam_idx = _primary_index(top_name)
        heatmap_disease = top_name
        if cam_idx is None:
            # top finding is not one the Grad-CAM model predicts (e.g. an
            # NIH-only finding when a CheXpert model is primary) — fall back to
            # that model's own strongest class so the overlay stays meaningful
            _own = _ml.LABEL_SETS[PRIMARY["label_set"]]
            cam_idx = int(np.argmax(
                [mean_probs[active_diseases.index(_ml.to_canonical(n))]
                 for n in _own]))
            # the heatmap now shows a DIFFERENT finding from the headline one;
            # the UI must say so rather than mislabel the image
            heatmap_disease = _ml.to_canonical(_own[cam_idx])
        # returned, not read from a global — safe under concurrent requests
        heatmap, heatmap_raw = generate_gradcam(img, cam_idx)
    except Exception as e:
        print(f"GradCAM error: {e}")

    # ── Zone localisation + abstention + provenance ───
    # Location changes meaning: upper-zone consolidation raises TB, lower-zone
    # raises bacterial pneumonia. And a finding predicted by only ONE model is
    # weaker evidence than one four models agree on — the UI must not present
    # them identically.
    zone = None
    try:
        if heatmap_raw is not None:
            zone = _cx.zone_of(heatmap_raw)
    except Exception as e:
        print(f"zone error: {e}")

    for p in predictions:
        d = p["disease"]
        p["single_source"] = d in SINGLE_SOURCE
        p["n_models"] = _coverage_count.get(d, 1)
        try:
            ab = _cx.should_abstain(p["probability"], p.get("uncertainty", 0), d)
            p["abstain"] = ab["abstain"]
            p["abstain_reason"] = ab["reason"]
            p["abstain_action"] = ab["action"]
        except Exception:
            p["abstain"] = False

    try:
        study_verdict = _cx.panel_abstention(predictions)
    except Exception:
        study_verdict = {"abstain": False, "message": None}

    if zone:
        try:
            zone["clinical_hint"] = _cx.zone_hint(
                zone, active_diseases[top_idx] if top_idx < len(active_diseases) else None)
        except Exception:
            pass

    # Top diseases for Stage 2
    #
    # NON-PATHOLOGIES MUST NEVER BE THE PRIMARY DIAGNOSIS.
    # "Support Devices" means a tube, line or pacemaker is visible; "No Finding"
    # means the film looks normal. Neither is a disease, neither has a treatment
    # plan, and neither should trigger an urgent-referral banner. Leaving them in
    # this list made the app announce "Support Devices Present — REFER TO
    # DISTRICT HOSPITAL URGENTLY" on a film whose actual finding was
    # infiltration. They are reported separately as clinical context instead.
    # RANKED BY SALIENCE, NOT RAW PROBABILITY.
    # Raw probability is not comparable across findings: Infiltration averages
    # 0.527 on every image while Cardiomegaly averages 0.074, so sorting by the
    # raw number put the ordinary finding first regardless of the film.
    # Salience asks "how unusual is this score FOR THIS FINDING", using each
    # one's measured baseline. Measured effect on 333 single-finding
    # radiologist-labelled films: top-1 30% -> 39%.
    #
    # The salience value is attached to every prediction and returned, so the
    # UI can explain the ordering. The displayed PROBABILITY is untouched.
    ranked = _cx.rank_by_salience(predictions, FINDING_BASELINES)
    _sal = {p["disease"]: p["salience"] for p in ranked}
    for p in predictions:
        p["salience"] = _sal.get(p["disease"])
        p["is_pathology"] = _mkext.is_pathology(p["disease"])

    # ── RELIABILITY GATE ───────────────────────────────────────────────────
    # Every finding is analysed, but only findings MEASURED at AUC >= 0.70 may
    # be reported as a diagnosis (core/finding_reliability.py). The rest are
    # withheld with an explicit message naming the finding and saying why.
    #
    # This exists because testing showed the system reporting findings it has
    # no basis for: Pleural Thickening measured AUC 0.497 with 99% sensitivity
    # and 1% specificity — it says "yes" to nearly every image — and it was
    # what flagged all 250 healthy chests as diseased. A tool that says "I
    # cannot assess this" is more use in a clinic than one that is right by
    # coincidence.
    for p in predictions:
        p["reliability"] = _rel.status(p["disease"])
        ev = _rel.evidence_for(p["disease"])
        p["validated_auc"] = ev["auc"]
        p["n_validated"] = ev["n_tested"]

    def _clears(p):
        t = DISEASE_THRESHOLDS.get(p["disease"], {})
        return p["probability"] >= t.get("detected", 0.50)

    diagnosable = [p for p in ranked
                   if _mkext.is_pathology(p["disease"])
                   and _rel.can_diagnose(p["disease"])]
    withheld = [p for p in ranked
                if _mkext.is_pathology(p["disease"])
                and not _rel.can_diagnose(p["disease"])]

    # Only findings that clear their own detection threshold count as present.
    # Threshold, not the looser 'borderline' — borderline exists to widen the
    # Stage-2 question net, not to assert a finding is there.
    flagged = [p for p in diagnosable if _clears(p)]

    # ── ONE PRIMARY DIAGNOSIS ──────────────────────────────────────────────
    # Previously the report listed four or five findings with percentages and
    # left the clinician to work out which mattered. The single most salient
    # finding that clears its threshold IS the answer; everything else is
    # supporting detail.
    primary = flagged[0] if flagged else None

    # ── WITHHELD NOTICES ───────────────────────────────────────────────────
    # If a finding we are NOT entitled to diagnose is nonetheless the strongest
    # signal on this film, say so rather than silently dropping it — otherwise a
    # patient with pleural thickening gets a clean report.
    withheld_notices = []
    for p in withheld[:3]:
        if p.get("salience", 0) >= 1.0 or _clears(p):
            msg = _rel.withhold_message(p["disease"])
            if msg:
                withheld_notices.append({**msg,
                                         "salience": p.get("salience"),
                                         "validated_auc": p.get("validated_auc")})

    # ── NORMAL ─────────────────────────────────────────────────────────────
    is_normal = primary is None
    normal_note = None
    if is_normal:
        if withheld_notices:
            normal_note = ("No finding that AyShCXR is validated to detect was "
                           "identified. However, see the note(s) below about "
                           "findings this system cannot assess.")
        else:
            normal_note = ("No abnormality detected among the findings AyShCXR "
                           "is validated to detect. IMPORTANT: a normal chest "
                           "X-ray does not exclude disease — early tuberculosis "
                           "in particular is often invisible on plain film. If "
                           "the patient has symptoms, investigate clinically "
                           "regardless of this result.")

    top_diseases = [p["disease"] for p in flagged][:4]

    context_findings = [
        {"disease": p["disease"], "probability": p["probability"],
         "salience": p.get("salience"),
         "kind": CANONICAL_FINDINGS.get(p["disease"], {}).get("kind", "disease")}
        for p in ranked
        if not _mkext.is_pathology(p["disease"]) and _clears(p)
    ]

    stage2_questions = get_stage2_questions_for_diseases(top_diseases)
    risk_notes, extra_diseases = get_risk_summary(
        age, gender, smoking, symptoms,
        occupation=occupation, conditions=conditions, medications=medications
    )

    return jsonify({
        "predictions"     : predictions,
        "top_diseases"    : top_diseases,
        "stage2_questions": stage2_questions,
        "heatmap"         : heatmap,
        "risk_notes"      : risk_notes,
        "extra_diseases"  : extra_diseases,
        "thresholds"      : DISEASE_THRESHOLDS,
        "model_name"      : model_name,
        "quality"         : quality,
        "zone"            : zone,
        "heatmap_disease" : heatmap_disease,
        "study_verdict"   : study_verdict,
        "context_findings": context_findings,
        "detected_risks"  : detected_risks,
        # ── the new, honest output layer ──
        "primary"         : primary,
        "is_normal"       : is_normal,
        "normal_note"     : normal_note,
        "withheld_notices": withheld_notices,
        "reliability"     : {
            "can_diagnose": _rel.summary()["diagnostic"],
            "withheld": _rel.summary()["unreliable"] + _rel.summary()["unvalidated"],
            "min_auc": _rel.MIN_DIAGNOSTIC_AUC,
        },
        "models_used"     : [e["id"] for e in LOADED],
        "calibrated"      : bool(_calib.load()),
        "patient"         : {
            "name": name, "age": age, "gender": gender,
            "duration": duration, "smoking": smoking,
            "occupation": occupation, "conditions": conditions,
            "medications": medications, "doctor": doctor,
            "patient_id": patient_id, "symptoms": symptoms
        }
    })


@app.route("/predict_stage2", methods=["POST"])
def predict_stage2():
    data           = request.json
    stage1         = data.get("stage1_data", {})
    stage2_answers = data.get("stage2_answers", {})
    predictions    = stage1.get("predictions", [])
    top_diseases   = stage1.get("top_diseases", [])

    predictions = apply_stage2_scores(predictions, stage2_answers, top_diseases)

    primary_disease = next((p["disease"] for p in predictions if p.get("is_primary")), None)
    advice          = []
    ai_detected     = set()

    for pred in predictions:
        thresh = DISEASE_THRESHOLDS.get(pred["disease"], {}).get("borderline", 0.35)
        show   = pred.get("is_primary") or (
            pred["disease"] in top_diseases and pred["probability"] > thresh
        )
        if show:
            report = get_disease_report(pred["disease"], pred["probability"])
            if report:
                report["uncertainty"] = pred["uncertainty"]
                advice.append(report)
                ai_detected.add(pred["disease"])

    extra_diseases   = stage1.get("extra_diseases", [])
    history_diseases = [d for d in extra_diseases if d not in ai_detected]
    history_advice   = []
    for disease in history_diseases:
        report = get_disease_report(disease, 0.0)
        if report:
            report["probability"]     = "Clinically suspected"
            report["urgency"]         = "medium"
            report["urgency_message"] = "Investigate based on clinical history"
            history_advice.append(report)

    return jsonify({
        "predictions"    : predictions,
        "top_diseases"   : top_diseases,
        "primary_disease": primary_disease,
        "advice"         : advice,
        "history_advice" : history_advice,
        "extra_diseases" : history_diseases,
        "heatmap"        : stage1.get("heatmap"),
        "risk_notes"     : stage1.get("risk_notes", []),
        "thresholds"     : DISEASE_THRESHOLDS,
        "model_name"     : model_name,
        "patient"        : stage1.get("patient", {})
    })


# ── Health check ────────────────────────────────────────────────────────────
# Needed on any deployed device: a way to ask "did the models actually load?"
# without running an inference. Returns 503 if the app is up but unusable, so a
# supervisor or the Jetson's own watchdog can restart it.
@app.route("/health")
def health():
    ok = bool(LOADED)
    return jsonify({
        "status": "ok" if ok else "degraded",
        "models_loaded": [e["id"] for e in LOADED],
        "models_requested": ENSEMBLE_IDS,
        "findings": num_diseases,
        "calibrated": bool(_calib.load()),
        "thresholds_measured": sum(1 for v in THRESHOLD_BASIS.values()
                                   if v == "measured"),
        "device": str(device),
    }), (200 if ok else 503)


if __name__ == "__main__":
    # ── SECURITY: debug mode and network binding ───────────────────────────
    # This previously ran `app.run(debug=True, host="0.0.0.0")`, which is the
    # most dangerous line in the file. debug=True serves the interactive
    # Werkzeug debugger on any unhandled exception, and that debugger allows
    # ARBITRARY CODE EXECUTION from the browser. Binding to 0.0.0.0 exposed it
    # to every machine on the network — on shared campus or hostel Wi-Fi, that
    # is remote code execution by anyone who can reach the port.
    #
    # Defaults are now safe: localhost only, no debugger. Both are opt-in via
    # environment variable, and enabling the debugger on a public interface is
    # refused outright rather than merely warned about.
    _debug = _os.environ.get("AYSHCXR_DEBUG", "0") == "1"
    _host  = _os.environ.get("AYSHCXR_HOST", "127.0.0.1")
    _port  = int(_os.environ.get("AYSHCXR_PORT", "5000"))

    if _debug and _host not in ("127.0.0.1", "localhost"):
        print()
        print("❌ Refusing to start: AYSHCXR_DEBUG=1 together with host "
              f"{_host!r} would expose the Werkzeug debugger — which allows "
              "remote code execution — to the whole network.")
        print("   Use one or the other, not both.")
        print()
        _sys.exit(2)

    print()
    print("=" * 62)
    print("  AyShCXR — Two-Stage Clinical Decision System")
    print("  by Subhrakant Sethi & Ayush Singh")
    print("=" * 62)
    print(f"  Models     : {len(LOADED)} loaded — {[e['id'] for e in LOADED]}")
    print(f"  Findings   : {num_diseases} canonical")
    print(f"  Grad-CAM   : {PRIMARY['id']} @ {IMG_SIZE}px")
    print(f"  Calibration: {'active' if _calib.load() else 'NOT LOADED'}")
    print(f"  Thresholds : {sum(1 for v in THRESHOLD_BASIS.values() if v == 'measured')}"
          f"/{len(THRESHOLD_BASIS)} measured against radiologist labels")
    print(f"  Symptoms   : Bayesian likelihood ratios (FiLM retired)")
    print(f"  Ranking    : salience vs per-finding baseline")
    print("-" * 62)
    print(f"  Debug mode : {'ON — DO NOT EXPOSE TO A NETWORK' if _debug else 'off'}")
    print(f"  Listening  : http://{_host}:{_port}")
    if _host == "0.0.0.0":
        print("  ⚠️  Bound to all interfaces — reachable by anyone on this network.")
    print("=" * 62)
    print()
    app.run(debug=_debug, host=_host, port=_port)
