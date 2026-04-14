# predict_single.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi
# Auto-detects: EfficientNet-B4 (main) or DenseNet-121 (backup)
# Features: uncertainty, risk factors, differential diagnosis,
# lab tests, follow-up, auto-save timestamped report

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from medical_knowledge import (
    get_disease_report, DISEASE_INFO, get_risk_summary
)

# ── Settings ──────────────────────────────────────────
IMG_SIZE  = 224
THRESHOLD = 0.5

DISEASES = [
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
]

# ── Device ────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── Transform ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# ── Load model — auto detect ─────────────────────────
def load_model():
    global DISEASES

    # ── Priority 1: EfficientNet-B4 ──────────────────
    if os.path.exists("efficientnet_b4_14class_best.pth"):
        print("Loading EfficientNet-B4 (main model)...")
        m        = models.efficientnet_b4(pretrained=False)
        old_conv = m.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding,
            bias=False
        )
        m.features[0][0] = new_conv
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f),
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 14)
        )
        m.load_state_dict(
            torch.load(
                "efficientnet_b4_14class_best.pth",
                map_location=device
            )
        )
        m.eval()
        m.to(device)
        print("✅ EfficientNet-B4 loaded (14 diseases)")
        return m, "EfficientNet-B4"

    # ── Priority 2: DenseNet-121 backup ──────────────
    elif os.path.exists("densenet121_14class_best.pth"):
        print("Loading DenseNet-121 (backup)...")
        m = models.densenet121(pretrained=False)
        m.features.conv0 = nn.Conv2d(
            1, 64, 7, 2, 3, bias=False
        )
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f),
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 14)
        )
        try:
            m.load_state_dict(
                torch.load(
                    "densenet121_14class_best.pth",
                    map_location=device
                )
            )
        except RuntimeError:
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, 14)
            )
            m.load_state_dict(
                torch.load(
                    "densenet121_14class_best.pth",
                    map_location=device
                )
            )
        m.eval()
        m.to(device)
        print("✅ DenseNet-121 loaded (14 diseases)")
        return m, "DenseNet-121"

    # ── Priority 3: ResNet-18 fallback ───────────────
    elif os.path.exists("resnet18_demo.pth"):
        print("⚠️  Using ResNet-18 fallback (4 diseases)")
        m = models.resnet18(pretrained=False)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.fc    = nn.Linear(512, 4)
        m.load_state_dict(
            torch.load("resnet18_demo.pth",
                       map_location=device)
        )
        DISEASES = [
            "Pneumonia", "Effusion",
            "Pneumothorax", "Infiltration"
        ]
        m.eval()
        m.to(device)
        return m, "ResNet-18"

    else:
        print("❌ No model found!")
        print("   Run build_and_train_demo.py first")
        exit()

# ── Monte Carlo Uncertainty ───────────────────────────
def predict_with_uncertainty(model, img_pil, n_passes=15):
    tensor = transform(img_pil).unsqueeze(0).to(device)

    def enable_dropout(m):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()

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

# ── GradCAM ───────────────────────────────────────────
def generate_gradcam(model, model_name, image_path,
                     target_idx):
    img    = Image.open(image_path).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    gradients   = []
    activations = []

    def save_grad(grad):
        gradients.append(grad)

    def hook_fn(module, inp, out):
        activations.append(out)
        out.register_hook(save_grad)

    # Correct hook per architecture
    if "EfficientNet" in model_name:
        hook_layer = model.features[-1]
    elif hasattr(model, "features") and hasattr(
        model.features, "denseblock4"
    ):
        hook_layer = model.features.denseblock4
    else:
        hook_layer = model.layer4

    handle = hook_layer.register_forward_hook(hook_fn)
    output = model(tensor)
    model.zero_grad()
    output[0, target_idx].backward()
    handle.remove()

    if not gradients or not activations:
        return None

    grad = gradients[0].squeeze().detach().cpu().numpy()
    act  = activations[0].squeeze().detach().cpu().numpy()

    if grad.ndim == 3:
        weights = grad.mean(axis=(1, 2))
        cam     = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]
    else:
        cam = grad

    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam /= cam.max()

    cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    img_rgb = np.array(
        img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    )
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        img_rgb, 0.5, heatmap, 0.5, 0
    )

    out_path = "patient_heatmap.png"
    cv2.imwrite(
        out_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )
    return out_path

# ── Collect patient info ──────────────────────────────
def collect_symptoms():
    print()
    print("=" * 65)
    print("   AyShCXR — PATIENT INFORMATION")
    print("=" * 65)

    name        = input("\nPatient name       : ").strip()
    age         = input("Age                : ").strip()
    gender      = input("Gender (M/F/Other) : ").strip()
    occupation  = input("Occupation         : ").strip()
    conditions  = input("Known conditions   : ").strip()
    medications = input("Current medications: ").strip()

    print("\n─── Symptom Checklist ───────────────────────")
    print("Answer y (yes) or n (no):\n")

    symptom_questions = [
        ("fever",         "Fever?"),
        ("cough",         "Cough?"),
        ("breathless",    "Shortness of breath?"),
        ("chest_pain",    "Chest pain?"),
        ("fatigue",       "Fatigue?"),
        ("weight_loss",   "Unexplained weight loss?"),
        ("night_sweats",  "Night sweats?"),
        ("haemoptysis",   "Coughing blood?"),
        ("wheezing",      "Wheezing?"),
        ("swelling",      "Leg or ankle swelling?"),
        ("orthopnoea",    "Cannot lie flat?"),
        ("pnd",           "Waking breathless at night?"),
        ("cyanosis",      "Bluish lips or skin?"),
        ("tb_contact",    "TB contact history?"),
        ("dust_exposure", "Dust or chemical exposure?"),
        ("heartburn",     "Heartburn or acid reflux?"),
        ("dysphagia",     "Difficulty swallowing?"),
        ("family_lung",   "Family history of lung disease?"),
    ]

    symptoms = {}
    for key, question in symptom_questions:
        while True:
            ans = input(
                f"  {question:<42} (y/n): "
            ).strip().lower()
            if ans in ["y", "yes"]:
                symptoms[key] = True
                break
            elif ans in ["n", "no"]:
                symptoms[key] = False
                break
            else:
                print("  Please enter y or n")

    duration = input("\nDuration of symptoms         : ").strip()
    smoking  = input(
        "Smoking history (yes/no/past): "
    ).strip().lower()
    symptoms["smoking"] = smoking in ["yes", "y"]

    return {
        "name": name, "age": age, "gender": gender,
        "occupation": occupation, "conditions": conditions,
        "medications": medications, "duration": duration,
        "smoking": smoking, "symptoms": symptoms
    }

# ── Generate full report ──────────────────────────────
def generate_report(patient_info, predictions,
                    uncertainties, heatmap_path,
                    model_name):

    name     = patient_info["name"]
    age      = patient_info["age"]
    gender   = patient_info["gender"]
    duration = patient_info["duration"]
    smoking  = patient_info["smoking"]
    symptoms = patient_info["symptoms"]
    now      = datetime.now().strftime("%d/%m/%Y %H:%M")

    detected = {
        d: p for d, p in predictions.items()
        if p > THRESHOLD
    }

    lines = []
    lines.append("=" * 65)
    lines.append("  AyShCXR — AI Chest X-Ray Clinical Report")
    lines.append("  by Subhrakant Sethi")
    lines.append(f"  Model     : {model_name}")
    lines.append(f"  Generated : {now}")
    lines.append("=" * 65)
    lines.append(f"Patient    : {name}")
    lines.append(f"Age/Gender : {age} / {gender}")
    lines.append(
        f"Occupation : "
        f"{patient_info['occupation'] or 'Not specified'}"
    )
    lines.append(
        f"Conditions : "
        f"{patient_info['conditions'] or 'None reported'}"
    )
    lines.append(
        f"Medications: "
        f"{patient_info['medications'] or 'None reported'}"
    )
    lines.append(f"Duration   : {duration}")
    lines.append(f"Smoking    : {smoking}")
    lines.append("-" * 65)

    # Risk factors + extra suspected diseases
    risk_notes, extra_diseases = get_risk_summary(
        age, gender, smoking, symptoms,
        occupation=patient_info.get("occupation",""),
        conditions=patient_info.get("conditions",""),
        medications=patient_info.get("medications","")
    )
    if risk_notes:
        lines.append("\n⚠  PATIENT RISK FACTORS:")
        for note in risk_notes:
            lines.append(f"   • {note}")

    if extra_diseases:
        lines.append(
            "\n🔍 ADDITIONALLY SUSPECTED FROM HISTORY:"
        )
        lines.append(
            "   (Not AI detected — based on occupation,"
            " conditions, medications, symptoms)"
        )
        for d in extra_diseases:
            lines.append(f"   ⚕ {d}")

    # Probability table
    lines.append("\nDISEASE PROBABILITY SCORES:")
    lines.append(
        f"{'Disease':<22} {'Prob':>7}  "
        f"{'±Unc':>6}  Status"
    )
    lines.append("-" * 55)

    for disease, prob in sorted(
        predictions.items(),
        key=lambda x: x[1], reverse=True
    ):
        pct    = prob * 100
        unc    = uncertainties.get(disease, 0) * 100
        status = "⚠ DETECTED" if prob > THRESHOLD \
                 else "✓ Clear"
        lines.append(
            f"{disease:<22} {pct:>6.1f}%  "
            f"±{unc:>4.1f}%  {status}"
        )

    # Symptoms reported
    active_syms = [
        k.replace("_", " ").title()
        for k, v in symptoms.items() if v
    ]
    if active_syms:
        lines.append("\nSYMPTOMS REPORTED:")
        for s in active_syms:
            lines.append(f"  • {s}")

    # Detailed findings
    if detected:
        lines.append("\n" + "=" * 65)
        lines.append("  DETAILED CLINICAL FINDINGS")
        lines.append("=" * 65)

        for disease, prob in sorted(
            detected.items(),
            key=lambda x: x[1], reverse=True
        ):
            report = get_disease_report(disease, prob)
            if not report:
                continue

            unc = uncertainties.get(disease, 0) * 100
            lines.append(f"\n{'─'*65}")
            lines.append(f"  {report['full_name']}")
            lines.append(
                f"  ICD-10: {report.get('icd10','—')}"
            )
            lines.append(
                f"  Probability  : {report['probability']}%"
                f" ± {unc:.1f}%"
            )
            lines.append(
                f"  Urgency      : "
                f"{report['urgency'].upper()}"
            )
            lines.append(
                f"  Action       : "
                f"{report['urgency_message']}"
            )
            lines.append(
                f"  Specialist   : {report['specialist']}"
            )
            lines.append(f"{'─'*65}")

            lines.append("\nDESCRIPTION:")
            lines.append(f"  {report['description']}")

            if report.get("imaging_findings"):
                lines.append("\nX-RAY FINDINGS:")
                lines.append(
                    f"  {report['imaging_findings']}"
                )

            if report.get("differential"):
                lines.append("\nDIFFERENTIAL DIAGNOSIS:")
                for d in report["differential"][:3]:
                    lines.append(f"  • {d}")

            if report.get("lab_tests"):
                lines.append("\nRECOMMENDED INVESTIGATIONS:")
                for t in report["lab_tests"][:4]:
                    lines.append(f"  • {t}")

            lines.append("\nTREATMENTS:")
            for t in report["treatments"][:5]:
                lines.append(f"  • {t}")

            lines.append("\nPRESCRIPTION GUIDANCE:")
            lines.append(f"  {report['prescription_note']}")

            if report.get("follow_up"):
                lines.append("\nFOLLOW-UP PLAN:")
                lines.append(f"  {report['follow_up']}")

            if report.get("admission_criteria"):
                lines.append("\nHOSPITAL ADMISSION IF:")
                lines.append(
                    f"  {report['admission_criteria']}"
                )

            lines.append(f"\n⚠  EMERGENCY SIGNS:")
            lines.append(f"  {report['emergency_signs']}")

    else:
        lines.append("\n" + "=" * 65)
        lines.append("✅ NO SIGNIFICANT FINDINGS DETECTED")
        lines.append(
            "   All probabilities below threshold."
        )
        lines.append(
            "   Consult a doctor if symptoms persist."
        )
        lines.append("=" * 65)

    # Smoking advisory
    if symptoms.get("smoking") or smoking in ["yes","y"]:
        lines.append("\n⚠  SMOKING ADVISORY:")
        lines.append(
            "   Smoking is the leading cause of lung cancer,"
            " COPD, and cardiovascular disease."
        )
        lines.append(
            "   Quitting now is the single most important"
            " step for lung health."
        )

    lines.append("\n" + "=" * 65)
    lines.append("⚠  DISCLAIMER")
    lines.append("=" * 65)
    lines.append(
        "This report is generated by AyShCXR, "
        "a research AI prototype."
    )
    lines.append(
        "It is NOT a substitute for professional "
        "medical diagnosis."
    )
    lines.append(
        "Always consult a qualified doctor or "
        "radiologist for clinical decisions."
    )
    lines.append("=" * 65)

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AyShCXR — AI Chest X-Ray Analysis")
    print("  by Subhrakant Sethi")
    print("=" * 65)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input(
            "\nEnter path to X-ray image: "
        ).strip().strip('"')

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    print(f"\n✅ Image: {image_path}")

    patient_info = collect_symptoms()

    print("\nLoading model...")
    model, model_name = load_model()
    print(f"✅ Active model: {model_name}")

    print("Analysing X-ray with uncertainty estimation...")
    img = Image.open(image_path).convert("L")
    mean_probs, std_probs = predict_with_uncertainty(
        model, img, n_passes=15
    )

    predictions   = {
        d: float(mean_probs[i])
        for i, d in enumerate(DISEASES)
    }
    uncertainties = {
        d: float(std_probs[i])
        for i, d in enumerate(DISEASES)
    }

    top_idx      = int(np.argmax(mean_probs))
    heatmap_path = None
    try:
        heatmap_path = generate_gradcam(
            model, model_name, image_path, top_idx
        )
        if heatmap_path:
            print(f"✅ Heatmap saved: {heatmap_path}")
    except Exception as e:
        print(f"⚠️  Heatmap failed: {e}")

    report = generate_report(
        patient_info, predictions,
        uncertainties, heatmap_path,
        model_name
    )

    print("\n" + report)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"ayshcxr_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved: {report_path}")

if __name__ == "__main__":
    main()