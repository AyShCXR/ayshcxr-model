# evaluate_localization.py
# AyShCXR — GradCAM Bounding Box Localization Evaluation
# by Subhrakant Sethi & Ayush Singh
# For Research Paper Results

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 65)
print("   AyShCXR — GRAD-CAM QUANTITATIVE LOCALIZATION EVALUATION")
print("   Metrics: Mean Intersection-over-Union (IoU) & Pointing Accuracy")
print("=" * 65)

# ── Device Setup ──────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Settings & Constants ──────────────────────────────
DISEASES = [
    "Atelectasis",    "Cardiomegaly",  "Effusion",
    "Infiltration",   "Mass",          "Nodule",
    "Pneumonia",      "Pneumothorax",  "Consolidation",
    "Edema",          "Emphysema",     "Fibrosis",
    "Pleural Thickening",              "Hernia"
]

# ── Model Auto-detection & Loading ───────────────────
def load_backbone():
    """Auto-detects and loads the image backbone."""
    SAFE_MODEL = "densenet121_BEST_auc0.8031_ep19_SAFE.pth"
    if os.path.exists(SAFE_MODEL):
        print(f"Loading model: {SAFE_MODEL}")
        m = models.densenet121(weights=None)
        m.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
            nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(p=0.3),   nn.Linear(512, 14)
        )
        m.load_state_dict(torch.load(SAFE_MODEL, map_location=device))
        return m, 224, "DenseNet-121"

    EFF_MODEL = "efficientnet_b4_14class_best.pth"
    if os.path.exists(EFF_MODEL):
        print(f"Loading model: {EFF_MODEL}")
        m = models.efficientnet_b4(weights=None)
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
        m.load_state_dict(torch.load(EFF_MODEL, map_location=device))
        return m, 380, "EfficientNet-B4"

    FALLBACK = "densenet121_14class_best.pth"
    if os.path.exists(FALLBACK):
        print(f"Loading model: {FALLBACK}")
        m = models.densenet121(weights=None)
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
        return m, 224, "DenseNet-121"

    print("❌ No valid image model weights found!")
    exit()

model, img_size, model_name = load_backbone()
model.eval()
model.to(device)

# ── Transform Setup ───────────────────────────────────
transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229])
])

# ── Bounding Box Data Loading ─────────────────────────
if not os.path.exists("BBox_List_2017.csv"):
    print("❌ Missing BBox_List_2017.csv in workspace!")
    exit()

# Parse the NIH bounding box list (header: Image Index, Finding Label, Bbox [x, y, w, h])
bbox_df = pd.read_csv("BBox_List_2017.csv", header=None, skiprows=1)
bbox_df.columns = ["Image Index", "Finding Label", "x", "y", "w", "h"]

# Resolve absolute image paths
def find_image_path(filename):
    for folder in os.listdir("."):
        if folder.startswith("images_"):
            path = os.path.join(folder, "images", filename)
            if os.path.exists(path):
                return path
    return None

bbox_df["resolved_path"] = bbox_df["Image Index"].apply(find_image_path)
bbox_df = bbox_df[bbox_df["resolved_path"].notna()].reset_index(drop=True)
print(f"Total resolved bounding boxes: {len(bbox_df):,}")

if len(bbox_df) == 0:
    print("❌ No bounding box images found in local folder. Check images_001/ through images_012/.")
    exit()

# ── GradCAM Core Hook ─────────────────────────────────
class GradCAM:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.gradients = []
        self.activations = []
        
        # Register hooks on correct last conv layer
        if "efficientnet" in model_name.lower():
            self.hook_layer = model.features[5]
        elif hasattr(model, "features") and hasattr(model.features, "denseblock4"):
            self.hook_layer = model.features.denseblock4
        else:
            self.hook_layer = model.features[-1]
            
        self.hook_layer.register_forward_hook(self.save_activation)
        self.hook_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def generate(self, tensor, target_class):
        self.gradients.clear()
        self.activations.clear()
        
        output = self.model(tensor)
        self.model.zero_grad()
        output[0, target_class].backward()
        
        if not self.gradients or not self.activations:
            return None
            
        grad = self.gradients[0].squeeze().detach().cpu().numpy()
        act = self.activations[0].squeeze().detach().cpu().numpy()
        
        if grad.ndim == 3:
            weights = grad.mean(axis=(1, 2))
            cam = np.zeros(act.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * act[i]
        else:
            cam = grad
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        return cam

gradcam = GradCAM(model, model_name)

# ── Evaluation Loop ───────────────────────────────────
print("\nStep 2 — Running Grad-CAM localization evaluation...")
results_list = []
cam_threshold = 0.5  # binarisation threshold

for idx in tqdm(range(len(bbox_df)), desc="Localizing"):
    row = bbox_df.iloc[idx]
    image_path = row["resolved_path"]
    disease = row["Finding Label"]
    
    if disease not in DISEASES:
        continue
    target_idx = DISEASES.index(disease)
    
    # Load original image and properties
    img_pil = Image.open(image_path).convert("L")
    w_orig, h_orig = img_pil.size  # usually 1024x1024
    
    # Run forward-backward through Grad-CAM
    tensor = transform(img_pil).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    cam = gradcam.generate(tensor, target_idx)
    
    if cam is None:
        continue
        
    # Resize activation map back to original image dimensions
    cam_orig = cv2.resize(cam, (w_orig, h_orig))
    
    # Binarise predicted mask
    pred_mask = (cam_orig >= cam_threshold).astype(np.uint8)
    
    # Define ground-truth mask from bounding box coordinates
    # CSV format: x, y, w, h
    x = int(round(row["x"]))
    y = int(round(row["y"]))
    w = int(round(row["w"]))
    h = int(round(row["h"]))
    
    gt_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    gt_mask[y:y+h, x:x+w] = 1
    
    # Calculate IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-8)
    
    # Calculate Pointing Accuracy: is max activation inside bbox?
    y_max, x_max = np.unravel_index(np.argmax(cam_orig), cam_orig.shape)
    pointing_acc = 1.0 if (x <= x_max < x + w) and (y <= y_max < y + h) else 0.0
    
    results_list.append({
        "Image": row["Image Index"],
        "Disease": disease,
        "IoU": iou,
        "PointingAccuracy": pointing_acc
    })

results_df = pd.DataFrame(results_list)

# ── Summary & Output ──────────────────────────────────
print("\n" + "=" * 65)
print("              SUMMARY OF GRAD-CAM LOCALIZATION RESULTS")
print("=" * 65)
print(f"{'Disease':<20} | {'BBoxes':<6} | {'Mean IoU':<10} | {'Pointing Acc':<12}")
print("-" * 60)

lines = []
lines.append("# Grad-CAM Localization Metrics Report\n")
lines.append(f"Evaluated Model: **{model_name}**")
lines.append(f"Heatmap Binarisation Threshold: **{cam_threshold}**\n")
lines.append(f"| Disease | BBoxes | Mean IoU | Pointing Acc |")
lines.append(f"|---|---|---|---|")

for disease in sorted(results_df["Disease"].unique()):
    df_d = results_df[results_df["Disease"] == disease]
    mean_iou = df_d["IoU"].mean()
    mean_pa = df_d["PointingAccuracy"].mean()
    count = len(df_d)
    
    print(f"{disease:<20} | {count:<6} | {mean_iou:<10.4f} | {mean_pa:<12.4%}")
    lines.append(f"| {disease} | {count} | {mean_iou:.4f} | {mean_pa:.2%} |")

# Overall Mean
mean_iou_all = results_df["IoU"].mean()
mean_pa_all = results_df["PointingAccuracy"].mean()
total_count = len(results_df)
print("-" * 60)
print(f"{'OVERALL MEAN':<20} | {total_count:<6} | {mean_iou_all:<10.4f} | {mean_pa_all:<12.4%}")
lines.append(f"| **OVERALL MEAN** | **{total_count}** | **{mean_iou_all:.4f}** | **{mean_pa_all:.2%}** |")
print("=" * 65)

# Save Report
out_file = "localization_results.md"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\n")
print(f"✅ Localization report saved to: {out_file}")
