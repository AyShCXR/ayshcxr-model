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

import os
import io
import json
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
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

app      = Flask(__name__)
IMG_SIZE = 224

DISEASES = [
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
]

# ── Per-disease optimal thresholds from calibration ──
# Derived from F1-optimised analysis on val_predictions_densenet.csv
# Infiltration threshold raised to 0.60 — was overconfident (mean 0.440)
DISEASE_THRESHOLDS = {
    "Atelectasis"      : {"detected": 0.45, "borderline": 0.32},
    "Cardiomegaly"     : {"detected": 0.52, "borderline": 0.36},
    "Effusion"         : {"detected": 0.50, "borderline": 0.35},
    "Infiltration"     : {"detected": 0.60, "borderline": 0.50},
    "Mass"             : {"detected": 0.51, "borderline": 0.36},
    "Nodule"           : {"detected": 0.49, "borderline": 0.34},
    "Pneumonia"        : {"detected": 0.44, "borderline": 0.30},
    "Pneumothorax"     : {"detected": 0.50, "borderline": 0.35},
    "Consolidation"    : {"detected": 0.52, "borderline": 0.37},
    "Edema"            : {"detected": 0.51, "borderline": 0.36},
    "Emphysema"        : {"detected": 0.57, "borderline": 0.40},
    "Fibrosis"         : {"detected": 0.42, "borderline": 0.28},
    "Pleural Thickening": {"detected": 0.45, "borderline": 0.30},
    "Hernia"           : {"detected": 0.47, "borderline": 0.32},
}

STAGE1_SYMPTOM_CATEGORIES = [
    { "category": "Respiratory", "symptoms": [
        {"id": "cough",       "label": "Cough"},
        {"id": "breathless",  "label": "Shortness of Breath"},
        {"id": "wheezing",    "label": "Wheezing"},
        {"id": "haemoptysis", "label": "Coughing Blood"},
        {"id": "dry_cough",   "label": "Dry Persistent Cough"},
        {"id": "sputum",      "label": "Productive Cough"}
    ]},
    { "category": "Chest", "symptoms": [
        {"id": "chest_pain",      "label": "Chest Pain"},
        {"id": "chest_tightness", "label": "Chest Tightness"},
        {"id": "pleuritic_pain",  "label": "Pain on Breathing"},
        {"id": "palpitations",    "label": "Heart Palpitations"}
    ]},
    { "category": "General", "symptoms": [
        {"id": "fever",          "label": "Fever"},
        {"id": "fatigue",        "label": "Fatigue"},
        {"id": "weight_loss",    "label": "Weight Loss"},
        {"id": "night_sweats",   "label": "Night Sweats"},
        {"id": "loss_appetite",  "label": "Loss of Appetite"},
        {"id": "confusion",      "label": "Confusion"}
    ]}
]

# ── Device ────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# ── X-Ray Validation ──────────────────────────────────
def validate_xray(img_pil):
    """
    Validates that the uploaded image is likely a chest X-ray.
    Returns: (is_valid: bool, reason: str)
    """
    img_gray = img_pil.convert("L")
    img_rgb  = img_pil.convert("RGB")
    width, height = img_pil.size

    # Check 1: Minimum size
    if width < 100 or height < 100:
        return False, "Image too small to be a chest X-ray (minimum 100x100 pixels)"

    # Check 2: Aspect ratio — X-rays are roughly square
    ratio = width / height
    if ratio < 0.5 or ratio > 2.0:
        return False, "Image aspect ratio is unusual for a chest X-ray"

    # Check 3: Colour channels — X-rays are grayscale
    stat_rgb = ImageStat.Stat(img_rgb)
    r_mean, g_mean, b_mean = stat_rgb.mean
    channel_diff = max(
        abs(r_mean - g_mean),
        abs(g_mean - b_mean),
        abs(r_mean - b_mean)
    )
    if channel_diff > 30:
        return False, (
            "Image appears to be a colour photograph, not an X-ray. "
            "Chest X-rays are grayscale. Please upload a chest radiograph."
        )

    # Check 4: Contrast — X-rays have bright bone and dark air
    stat_gray = ImageStat.Stat(img_gray)
    std_brightness = stat_gray.stddev[0]
    if std_brightness < 20:
        return False, "Image has insufficient contrast for an X-ray."

    # Check 5: Overexposed natural photos
    mean_brightness = stat_gray.mean[0]
    if mean_brightness > 220:
        return False, "Image appears overexposed. Please upload a chest radiograph."

    # Check 6: Colour saturation — X-rays have near-zero saturation
    pixels = np.array(img_rgb, dtype=np.float32)
    r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    saturation = np.where(maxc > 0, (maxc - minc) / maxc, 0)
    if saturation.mean() > 0.15:
        return False, (
            "Image appears to be a colour photograph. "
            "Please upload a grayscale chest radiograph (PA or AP view)."
        )

    return True, "Valid X-ray image"


# ── Load model ────────────────────────────────────────
def load_model():
    """
    v6 FIX: Explicitly loads the confirmed 0.8031 safe model.
    Priority order:
      1. densenet121_BEST_auc0.8031_ep19_SAFE.pth — confirmed best
      2. densenet121_14class_best.pth              — fallback
      3. efficientnet_b4_14class_best.pth          — if available
    Never auto-loads the bad 0.7830 model (densenet121_14class.pth)
    """

    # Priority 1 — explicitly named safe model
    SAFE_MODEL = "densenet121_BEST_auc0.8031_ep19_SAFE.pth"
    if os.path.exists(SAFE_MODEL):
        print(f"✅ Loading confirmed best model: {SAFE_MODEL}")
        m = models.densenet121(pretrained=False)
        m.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
            nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(p=0.3),   nn.Linear(512, 14)
        )
        try:
            m.load_state_dict(torch.load(SAFE_MODEL, map_location=device))
            print(f"✅ DenseNet-121 loaded — AUC 0.8031 confirmed")
            return m, 14, DISEASES, "DenseNet-121 (AUC 0.8031)"
        except RuntimeError:
            print("⚠️  Trying old 2-layer head...")
            m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, 14))
            m.load_state_dict(torch.load(SAFE_MODEL, map_location=device))
            print(f"✅ DenseNet-121 loaded (old arch)")
            return m, 14, DISEASES, "DenseNet-121 (AUC 0.8031)"

    # Priority 2 — EfficientNet if available
    EFFNET_MODEL = "efficientnet_b4_14class_best.pth"
    if os.path.exists(EFFNET_MODEL):
        print(f"✅ Loading EfficientNet-B4: {EFFNET_MODEL}")
        m        = models.efficientnet_b4(pretrained=False)
        old_conv = m.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, old_conv.kernel_size,
                             old_conv.stride, old_conv.padding, bias=False)
        m.features[0][0] = new_conv
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
            nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(p=0.3),   nn.Linear(512, 14)
        )
        m.load_state_dict(torch.load(EFFNET_MODEL, map_location=device))
        print(f"✅ EfficientNet-B4 loaded")
        return m, 14, DISEASES, "EfficientNet-B4"

    # Priority 3 — fallback to densenet best
    FALLBACK = "densenet121_14class_best.pth"
    if os.path.exists(FALLBACK):
        print(f"⚠️  Safe model not found. Loading fallback: {FALLBACK}")
        m = models.densenet121(pretrained=False)
        m.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
            nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(p=0.3),   nn.Linear(512, 14)
        )
        try:
            m.load_state_dict(torch.load(FALLBACK, map_location=device))
        except RuntimeError:
            m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, 14))
            m.load_state_dict(torch.load(FALLBACK, map_location=device))
        print(f"✅ DenseNet-121 fallback loaded")
        return m, 14, DISEASES, "DenseNet-121"

    print("❌ No model found! Place densenet121_BEST_auc0.8031_ep19_SAFE.pth in C:\\AyShCXR\\")
    exit()

model, num_diseases, active_diseases, model_name = load_model()
model.eval()
model.to(device)

# ── GradCAM ───────────────────────────────────────────
def generate_gradcam(img_pil, target_class):
    tensor    = transform(img_pil).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    gradients = []
    activations = []

    def save_grad(grad):   gradients.append(grad)
    def hook_fn(m, i, o):
        activations.append(o)
        o.register_hook(save_grad)

    # v6 FIX: EfficientNet hook corrected features[6] → features[5]
    if isinstance(model, models.EfficientNet):
        hook_layer = model.features[5]   # ← FIXED (was features[6])
    elif hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        hook_layer = model.features.denseblock4  # confirmed correct for DenseNet
    elif hasattr(model, "layer4"):
        hook_layer = model.layer4
    else:
        hook_layer = model.features[-1]

    handle = hook_layer.register_forward_hook(hook_fn)
    model.eval()
    output = model(tensor)
    model.zero_grad()
    output[0, target_class].backward()
    handle.remove()

    if not gradients or not activations:
        return None

    grad = gradients[0].squeeze().detach().cpu().numpy()
    act  = activations[0].squeeze().detach().cpu().numpy()

    if grad.ndim == 3 and act.ndim == 3:
        weights = np.mean(grad, axis=(1, 2))
        cam     = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]
    else:
        cam = grad if grad.ndim == 2 else grad.mean(0)

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = (cam - cam.min()) / cam.max()

    cam     = cv2.GaussianBlur(cam, (5, 5), 0)
    cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    img_rgb = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ── Disease dependency correction ─────────────────────
def apply_disease_dependencies(predictions):
    """
    Corrects Infiltration overconfidence bias.
    When cardiac disease is detected, perihilar haziness
    is a secondary feature — not standalone Infiltration.
    Infiltration mean score was 0.440 across ALL images —
    clearly biased. This correction suppresses it when
    cardiac disease is the more likely explanation.
    """
    pred_dict = {p["disease"]: p["probability"] for p in predictions}

    cardiac_score = max(
        pred_dict.get("Cardiomegaly", 0),
        pred_dict.get("Edema", 0),
        pred_dict.get("Effusion", 0)
    )
    if cardiac_score > 0.35 and pred_dict.get("Infiltration", 0) > 0.40:
        suppression = 0.5 + (cardiac_score - 0.35) * 0.3
        pred_dict["Infiltration"] = round(pred_dict["Infiltration"] * suppression, 4)
        dominant = max(["Cardiomegaly","Edema","Effusion"], key=lambda d: pred_dict.get(d, 0))
        pred_dict[dominant] = round(min(pred_dict[dominant] * 1.15, 0.99), 4)

    if pred_dict.get("Consolidation", 0) > 0.45:
        pred_dict["Pneumonia"] = round(min(pred_dict.get("Pneumonia", 0) * 1.15, 0.99), 4)

    if pred_dict.get("Emphysema", 0) > 0.50:
        pred_dict["Pneumothorax"] = round(min(pred_dict.get("Pneumothorax", 0) * 1.10, 0.99), 4)

    if pred_dict.get("Effusion", 0) > 0.55:
        pred_dict["Cardiomegaly"] = round(min(pred_dict.get("Cardiomegaly", 0) * 1.10, 0.99), 4)

    return [{**p, "probability": pred_dict[p["disease"]]} for p in predictions]

# ── History boost ─────────────────────────────────────
def apply_history_boost(predictions, symptoms, occupation, conditions):
    boosted = []
    occ  = occupation.lower() if occupation else ""
    cond = conditions.lower() if conditions else ""
    for pred in predictions:
        prob    = pred["probability"]
        disease = pred["disease"]
        boost   = 0.0
        if disease == "Fibrosis":
            if any(w in occ for w in ["mine","coal","asbestos","quarry","sandblast"]): boost += 0.08
            if any(w in occ for w in ["welder","metal","foundry"]): boost += 0.05
        if disease == "Pneumonia":
            if any(w in cond for w in ["diabetes","dm","sugar"]) and symptoms.get("fever"): boost += 0.07
            if any(w in cond for w in ["hiv","aids","immunocompromised"]): boost += 0.08
        if disease == "Effusion":
            if any(w in cond for w in ["heart failure","ccf","chf","cardiac"]): boost += 0.08
            if symptoms.get("orthopnoea") and symptoms.get("swelling"): boost += 0.06
        if disease == "Cardiomegaly":
            if any(w in cond for w in ["hypertension","high bp","htn"]): boost += 0.07
        if disease == "Edema":
            if any(w in cond for w in ["heart failure","ccf","renal","nephrotic"]): boost += 0.07
        if disease == "Emphysema":
            if any(w in occ for w in ["mine","coal","textile","cotton"]): boost += 0.07
        if disease == "Infiltration":
            if symptoms.get("tb_contact") and symptoms.get("night_sweats"): boost += 0.08
        if disease == "Mass":
            if symptoms.get("haemoptysis"): boost += 0.07
        if disease == "Atelectasis":
            if symptoms.get("recent_surgery"): boost += 0.08
        boost    = min(boost, 0.10)
        new_pred = dict(pred)
        new_pred["probability"]   = round(min(prob + boost, 0.99), 4)
        new_pred["boost_applied"] = round(boost, 4)
        boosted.append(new_pred)
    return boosted

# ── MC Dropout ────────────────────────────────────────
def predict_with_uncertainty(img_pil, n_passes=20):
    """
    v6 FIX: n_passes increased from 10 → 20 for more stable
    uncertainty estimates. More passes = less variance in
    the uncertainty score itself.
    """
    tensor = transform(img_pil).unsqueeze(0).to(device)
    def enable_dropout(m):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout): mod.train()
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            out  = model(tensor)
            prob = torch.sigmoid(out).squeeze()
            preds.append(prob.cpu().numpy())
    preds = np.array(preds)
    model.eval()
    return preds.mean(axis=0), preds.std(axis=0)

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
  --bg:#060812;--surface:#0d1117;--glass:rgba(255,255,255,0.04);
  --glass-border:rgba(255,255,255,0.08);--accent:#00d4ff;--accent2:#0099cc;
  --gold:#f5a623;--gold2:#c47f28;--danger:#ff4757;--warn:#ffa502;
  --success:#2ed573;--purple:#7c3aed;--purple2:#a78bfa;
  --text:#f0f4f8;--muted:#64748b;--dim:#1e2530;
  --mono:'JetBrains Mono',monospace;--sans:'Instrument Sans',sans-serif;
  --display:'Syne',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden;}
.orb{position:fixed;border-radius:50%;filter:blur(120px);pointer-events:none;z-index:0;opacity:0.15;animation:orbFloat 8s ease-in-out infinite;}
.orb-1{width:600px;height:600px;background:#00d4ff;top:-200px;left:-200px;}
.orb-2{width:500px;height:500px;background:#7c3aed;bottom:-100px;right:-100px;animation-delay:-4s;}
.orb-3{width:300px;height:300px;background:#f5a623;top:40%;left:40%;animation-delay:-2s;}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1);}50%{transform:translate(30px,20px) scale(1.05);}}
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
.hero-tag{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10px;color:var(--accent);letter-spacing:2px;text-transform:uppercase;padding:6px 16px;border:1px solid rgba(0,212,255,0.2);border-radius:20px;background:rgba(0,212,255,0.05);margin-bottom:24px;}
.hero-tag::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);}
.hero h2{font-family:var(--display);font-size:64px;font-weight:800;line-height:1.0;margin-bottom:20px;background:linear-gradient(135deg,#ffffff 0%,#00d4ff 50%,#7c3aed 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero p{font-size:16px;color:var(--muted);max-width:580px;margin:0 auto 32px;line-height:1.7;font-weight:300;}

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
.gcard{background:var(--glass);border:1px solid var(--glass-border);border-radius:20px;padding:28px;backdrop-filter:blur(20px);position:relative;overflow:hidden;transition:transform 0.3s ease,box-shadow 0.3s ease;}
.gcard::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);}
.gcard:hover{transform:translateY(-2px);box-shadow:0 20px 60px rgba(0,0,0,0.4);}
.gcard.glow-cyan{box-shadow:0 0 40px rgba(0,212,255,0.08);}
.gcard.glow-purple{box-shadow:0 0 40px rgba(124,58,237,0.08);}
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
.symptom-cat{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin:14px 0 8px;}
.symptoms-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.symptom-item{display:flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid var(--glass-border);border-radius:8px;cursor:pointer;transition:all 0.2s ease;font-size:12px;background:rgba(255,255,255,0.02);}
.symptom-item:hover{border-color:rgba(0,212,255,0.3);background:rgba(0,212,255,0.04);}
.symptom-item input[type="checkbox"]{width:14px;height:14px;accent-color:var(--accent);cursor:pointer;flex-shrink:0;}
.symptom-item.checked{border-color:rgba(0,212,255,0.4);background:rgba(0,212,255,0.08);color:var(--accent);}

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
    <span class="hdr-pill">14 Diseases · v6</span>
  </div>
</header>

<main>

  <!-- HERO -->
  <div class="hero">
    <div class="hero-tag">Clinical AI Research System</div>
    <h2 data-en>AI-Assisted<br>Pulmonary Screening</h2>
    <h2 data-hi style="display:none">AI-सहायक<br>फुफ्फुसीय जांच</h2>
    <p data-en>Upload a chest radiograph. Our two-stage AI narrows 14 diseases to the most likely findings, then uses targeted clinical questions to identify one primary diagnosis.</p>
    <p data-hi style="display:none">छाती का एक्स-रे अपलोड करें। हमारी दो-चरण AI 14 बीमारियों को सबसे संभावित निष्कर्षों तक संकुचित करती है, फिर एक प्राथमिक निदान की पहचान करती है।</p>
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
      <p style="font-size:12px;color:var(--muted);margin-bottom:14px;line-height:1.6">
        <span data-en>Select all symptoms currently present. These narrow 14 diseases to the most likely 3-4 findings.</span>
        <span data-hi style="display:none">सभी मौजूदा लक्षण चुनें। ये 14 बीमारियों को 3-4 सबसे संभावित तक संकुचित करते हैं।</span>
      </p>
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
        <span data-en>Disease Probability Scores</span>
        <span data-hi style="display:none">रोग संभाव्यता स्कोर</span>
        <span style="font-size:9px;color:var(--muted);margin-left:4px">(AI · calibrated · mean ± uncertainty)</span>
      </div>
      <div class="viz-tabs">
        <div class="viz-tab active" onclick="switchViz('bars',this)" data-en>Bar Chart</div>
        <div class="viz-tab" onclick="switchViz('radar',this)" data-en>Radar Chart</div>
      </div>
      <div id="bars-view"><div id="prob-bars"></div></div>
      <div id="radar-view">
        <div class="radar-wrap">
          <canvas id="radarCanvas" width="500" height="400"></canvas>
        </div>
        <div class="radar-legend" id="radar-legend"></div>
      </div>
    </div>

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
  {category:"Respiratory",categoryHi:"श्वसन",symptoms:[
    {id:"cough",label:"Cough",labelHi:"खांसी"},
    {id:"breathless",label:"Shortness of Breath",labelHi:"सांस फूलना"},
    {id:"wheezing",label:"Wheezing",labelHi:"घरघराहट"},
    {id:"haemoptysis",label:"Coughing Blood",labelHi:"खून खांसी"},
    {id:"dry_cough",label:"Dry Persistent Cough",labelHi:"सूखी खांसी"},
    {id:"sputum",label:"Productive Cough",labelHi:"बलगम वाली खांसी"}
  ]},
  {category:"Chest",categoryHi:"छाती",symptoms:[
    {id:"chest_pain",label:"Chest Pain",labelHi:"छाती दर्द"},
    {id:"chest_tightness",label:"Chest Tightness",labelHi:"छाती में जकड़न"},
    {id:"pleuritic_pain",label:"Pain on Breathing",labelHi:"सांस में दर्द"},
    {id:"palpitations",label:"Heart Palpitations",labelHi:"दिल की धड़कन"}
  ]},
  {category:"General",categoryHi:"सामान्य",symptoms:[
    {id:"fever",label:"Fever",labelHi:"बुखार"},
    {id:"fatigue",label:"Fatigue",labelHi:"थकान"},
    {id:"weight_loss",label:"Weight Loss",labelHi:"वजन घटना"},
    {id:"night_sweats",label:"Night Sweats",labelHi:"रात को पसीना"},
    {id:"loss_appetite",label:"Loss of Appetite",labelHi:"भूख न लगना"},
    {id:"confusion",label:"Confusion",labelHi:"भ्रम"}
  ]}
];

const container=document.getElementById("symptoms-container");
STAGE1_CATEGORIES.forEach(cat=>{
  const catLabel=document.createElement("div");
  catLabel.className="symptom-cat";
  catLabel.innerHTML=`<span data-en>${cat.category}</span><span data-hi style="display:none">${cat.categoryHi}</span>`;
  container.appendChild(catLabel);
  const grid=document.createElement("div");
  grid.className="symptoms-grid";
  cat.symptoms.forEach(s=>{
    const lbl=document.createElement("label");
    lbl.className="symptom-item";
    lbl.innerHTML=`<input type="checkbox" id="sym-${s.id}" value="${s.id}">
      <span data-en>${s.label}</span><span data-hi style="display:none">${s.labelHi}</span>`;
    lbl.querySelector("input").addEventListener("change",e=>{
      lbl.classList.toggle("checked",e.target.checked);
    });
    grid.appendChild(lbl);
  });
  container.appendChild(grid);
});

// ═══ FILE HANDLING ═══
let selectedFile=null,stage1Data=null,stage2Qs=[],stage2Answers={},reportData=null;
let isCameraCapture=false;

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
    checked[s.id]=document.getElementById(`sym-${s.id}`).checked;
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

  // Probability bars
  const barsDiv=document.getElementById("prob-bars");
  barsDiv.innerHTML="";
  const thresholds=data.thresholds||{};
  data.predictions.forEach(item=>{
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
    barsDiv.innerHTML+=`
      <div class="disease-row">
        <div class="disease-hdr">
          <span class="disease-name">${item.disease}${badgeHtml}</span>
          <span class="disease-right">${uncHtml}<span class="disease-pct" style="color:${color}">${pct}%</span></span>
        </div>
        <div class="bar-track"><div class="bar-fill ${barClass}" style="width:${Math.min(parseFloat(pct),100)}%"></div></div>
      </div>`;
  });

  // Store for radar
  radarData=data.predictions;

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
  text+="ALL DISEASE SCORES:\n"+"-".repeat(50)+"\n";
  reportData.predictions.forEach(p2=>{
    const score=(p2.stage2_score||p2.probability)*100;
    const status=p2.is_primary?"🎯 PRIMARY":score>50?"⚠ DETECTED":score>35?"~ BORDERLINE":"✓ Clear";
    text+=p2.disease.padEnd(22)+score.toFixed(1).padStart(6)+"%  "+status+"\n";
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
  ["pt-name","pt-age","pt-duration","pt-notes","pt-occupation","pt-conditions","pt-medications","pt-doctor","pt-id"].forEach(id=>{const el=document.getElementById(id);if(el)el.value="";});
  ["risk-summary-box","extra-diseases-box","primary-finding-box","other-findings-box"].forEach(id=>{document.getElementById(id).style.display="none";});
  setStage(1);
  window.scrollTo({top:0,behavior:"smooth"});
}

// Model info
fetch("/model_info").then(r=>r.json()).then(data=>{
  document.getElementById("model-badge").textContent=data.model_name;
}).catch(()=>document.getElementById("model-badge").textContent="AyShCXR v6");
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
    return jsonify({"model_name": model_name})

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
    symptoms   = json.loads(request.form.get("symptoms", "{}"))

    # ── X-ray validation ──────────────────────────────
    try:
        img_bytes = file.stream.read()
        file.stream.seek(0)
        img_pil = Image.open(io.BytesIO(img_bytes))
        is_valid, reason = validate_xray(img_pil)
        if not is_valid:
            return jsonify({"validation_failed": True, "error": reason}), 200
        img = img_pil.convert("L")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {str(e)}"}), 400

    # ── AI prediction ─────────────────────────────────
    try:
        mean_probs, std_probs = predict_with_uncertainty(img, n_passes=20)
    except Exception:
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.sigmoid(output).squeeze()
        mean_probs = probs.cpu().numpy()
        std_probs  = np.zeros_like(mean_probs)

    predictions = [
        {"disease": d, "probability": round(float(mean_probs[i]), 4),
         "uncertainty": round(float(std_probs[i]), 4)}
        for i, d in enumerate(active_diseases)
    ]

    # Apply dependency correction then history boost
    predictions = apply_disease_dependencies(predictions)
    predictions = apply_history_boost(predictions, symptoms, occupation, conditions)

    # GradCAM
    top_idx = int(np.argmax(mean_probs))
    heatmap = None
    try:
        heatmap = generate_gradcam(img, top_idx)
    except Exception as e:
        print(f"GradCAM error: {e}")

    # Top diseases for Stage 2
    sorted_preds = sorted(predictions, key=lambda x: x["probability"], reverse=True)
    top_diseases = [p["disease"] for p in sorted_preds if p["probability"] >= 0.25][:4]
    if not top_diseases:
        top_diseases = [p["disease"] for p in sorted_preds[:3]]

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


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  AyShCXR v6 — Two-Stage Clinical Decision System")
    print("  by Subhrakant Sethi & Ayush Singh")
    print(f"  Active model  : {model_name}")
    print("  v6 fixes:")
    print("    ✅ Explicit model loading — 0.8031 safe model")
    print("    ✅ MC Dropout 10 → 20 passes")
    print("    ✅ EfficientNet GradCAM hook fixed")
    print("    ✅ X-ray validation (rejects selfies/photos)")
    print("    ✅ Per-disease calibrated thresholds")
    print("    ✅ Disease dependency correction")
    print("    ✅ GradCAM smoothed + correct DenseNet hook")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 55)
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)