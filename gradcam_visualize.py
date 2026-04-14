# gradcam_visualize.py
# Updated for DenseNet-121 + 14 diseases
# Generates GradCAM heatmap for any chest X-ray

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import argparse
import os

# ── Device ────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

IMG_SIZE = 224

# ── All 14 diseases ───────────────────────────────────
DISEASES = [
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
]

# ── Transform ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# ── Load model ────────────────────────────────────────
def load_model():
    """
    Loads best available model.
    Tries DenseNet-121 first then falls back to ResNet-18
    """
    # Try new DenseNet-121 model first
    if os.path.exists("densenet121_14class_best.pth"):
        print("Loading DenseNet-121 (14 diseases)...")
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(
            1, 64, 7, 2, 3, bias=False
        )
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f),
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 14)
        )
        model.load_state_dict(
            torch.load(
                "densenet121_14class_best.pth",
                map_location=device
            )
        )
        print("✅ DenseNet-121 loaded!")
        return model, DISEASES, "densenet"

    # Fallback to old ResNet-18
    elif os.path.exists("resnet18_demo.pth"):
        print("⚠️  Using ResNet-18 fallback (4 diseases)")
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(
            1, 64, 7, 2, 3, bias=False
        )
        model.fc = nn.Linear(512, 4)
        model.load_state_dict(
            torch.load(
                "resnet18_demo.pth",
                map_location=device
            )
        )
        diseases = [
            "Pneumonia", "Effusion",
            "Pneumothorax", "Infiltration"
        ]
        return model, diseases, "resnet"

    else:
        print("❌ No model file found!")
        print("   Run build_and_train_demo.py first")
        exit()

# ── GradCAM ───────────────────────────────────────────
def generate_gradcam(model, tensor, target_class,
                     model_type):
    """
    Generates GradCAM heatmap for the target class.
    Works with both DenseNet-121 and ResNet-18.
    """
    gradients   = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def hook_fn(module, inp, out):
        activations.append(out)
        out.register_hook(save_gradient)

    # Hook the correct last conv layer
    if model_type == "densenet":
        # DenseNet-121 last conv block
        hook_layer = model.features.denseblock4
    else:
        # ResNet-18 last residual block
        hook_layer = model.layer4

    handle = hook_layer.register_forward_hook(hook_fn)

    # Forward pass
    output = model(tensor)
    model.zero_grad()
    output[0, target_class].backward()
    handle.remove()

    if not gradients or not activations:
        print("⚠️  GradCAM hooks failed")
        return None

    grad = gradients[0].squeeze().detach().cpu().numpy()
    act  = activations[0].squeeze().detach().cpu().numpy()

    # Handle different tensor dimensions
    if grad.ndim == 3:
        weights = grad.mean(axis=(1, 2))
        cam     = np.zeros(
            act.shape[1:], dtype=np.float32
        )
        for i, w in enumerate(weights):
            cam += w * act[i]
    elif grad.ndim == 2:
        cam = grad
    else:
        cam = grad.mean(axis=0)

    # Normalise
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()

    # Resize to image size
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam

# ── Create overlay ────────────────────────────────────
def create_overlay(img_pil, cam):
    """
    Overlays GradCAM heatmap on original X-ray image
    """
    img_rgb  = np.array(
        img_pil.resize(
            (IMG_SIZE, IMG_SIZE)
        ).convert("RGB")
    )
    heatmap  = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(
        img_rgb, 0.5, heatmap, 0.5, 0
    )
    return overlay

# ── Main ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GradCAM visualisation for chest X-ray"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to chest X-ray image"
    )
    parser.add_argument(
        "--out", default="cam_overlay.png",
        help="Output heatmap path (default: cam_overlay.png)"
    )
    parser.add_argument(
        "--disease", default=None,
        help="Disease to visualise (default: top prediction)"
    )
    args = parser.parse_args()

    # Check image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return

    # Load model
    model, diseases, model_type = load_model()
    model.eval()
    model.to(device)

    # Load and transform image
    print(f"\nLoading image: {args.image}")
    img    = Image.open(args.image).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    # Get predictions
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.sigmoid(output).squeeze()

    # Show all predictions
    print("\n─── PREDICTIONS ───────────────────────────")
    print(f"{'Disease':<22} {'Probability':>12}   Status")
    print("─" * 50)

    detected = []
    for i, (disease, prob) in enumerate(
        zip(diseases, probs)
    ):
        p      = prob.item()
        status = "⚠️  DETECTED" if p > 0.5 else "✅ Clear"
        bar    = "█" * int(p * 20)
        print(f"{disease:<22} {p:>8.1%}   {bar}  {status}")
        if p > 0.5:
            detected.append((i, disease, p))

    # Determine target class for GradCAM
    if args.disease:
        if args.disease in diseases:
            target_idx = diseases.index(args.disease)
            target_name = args.disease
            print(f"\nVisualising: {target_name} (specified)")
        else:
            print(f"⚠️  '{args.disease}' not found")
            print(f"   Available: {', '.join(diseases)}")
            target_idx  = int(probs.argmax().item())
            target_name = diseases[target_idx]
    else:
        target_idx  = int(probs.argmax().item())
        target_name = diseases[target_idx]
        print(f"\nVisualising: {target_name} (top prediction)")

    # Generate GradCAM
    print("\nGenerating GradCAM heatmap...")
    tensor_grad = transform(img).unsqueeze(0).to(device)
    tensor_grad.requires_grad_(True)

    cam = generate_gradcam(
        model, tensor_grad,
        target_idx, model_type
    )

    if cam is None:
        print("❌ GradCAM generation failed")
        return

    # Create and save overlay
    overlay = create_overlay(img, cam)
    output_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, output_bgr)

    print(f"\n✅ Heatmap saved → {args.out}")
    print(f"   Target disease : {target_name}")
    print(f"   Probability    : {probs[target_idx]:.1%}")

    # Summary
    print("\n─── SUMMARY ───────────────────────────────")
    if detected:
        print("Detected diseases:")
        for idx, name, prob in sorted(
            detected, key=lambda x: x[2], reverse=True
        ):
            print(f"  ⚠️  {name:<22} {prob:.1%}")
    else:
        print("✅ No diseases detected above threshold")

    print(f"\nHeatmap shows where model focused for:")
    print(f"  → {target_name}")
    print(f"\nTo visualise a different disease use:")
    print(f"  python gradcam_visualize.py "
          f"--image {args.image} "
          f"--disease Pneumonia")

if __name__ == "__main__":
    main()