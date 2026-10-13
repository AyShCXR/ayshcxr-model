# eval_film_simulation.py
# AyShCXR — FiLM Test Set Simulation & Evaluation
# by Subhrakant Sethi & Ayush Singh
# For Research Paper Results


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
# ── end path bootstrap ──────────────────────────────────────────────────────

import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

print("=" * 65)
print("   AyShCXR — FiLM SIMULATED EHR TEST-SET EVALUATION")
print("   Ablations: Coherent, Noisy (10%/20%/30%), Contradictory")
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

SYMPTOM_COLS = [
    "breathless_present", "breathless_mild", "breathless_moderate", "breathless_severe",
    "cough_dry", "cough_productive", "cough_bloodstreaked",
    "fever_low", "fever_high",
    "haemoptysis_present", "haemoptysis_frank",
    "sputum_present", "sputum_purulent",
    "fatigue_present", "fatigue_severe",
    "smoking_present", "smoking_heavy",
    "chest_pain", "chest_tightness", "pleuritic_pain",
    "wheezing", "palpitations", "swelling",
    "tb_contact", "recent_surgery", "night_sweats", "weight_loss",
    "fever_dur_short", "fever_dur_medium", "fever_dur_long",
    "cough_dur_short", "cough_dur_medium", "cough_dur_long",
    "breathless_dur_acute", "breathless_dur_subacute", "breathless_dur_chronic",
]
SYMPTOM_DIM = len(SYMPTOM_COLS)  # 36

# ── Import Clinical Prevalences ───────────────────────
try:
    from medical_knowledge import SYMPTOM_PREVALENCE_BY_DISEASE
except ImportError:
    print("❌ Could not import medical_knowledge.py!")
    exit()

# ── FiLM Architecture Definitions ──────────────────────
class _SymptomEncoder(nn.Module):
    def __init__(self, symptom_dim=36, feature_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(symptom_dim, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(256, 512),         nn.ReLU(inplace=True),
        )
        self.gamma_head = nn.Linear(512, feature_dim)
        self.beta_head  = nn.Linear(512, feature_dim)
    def forward(self, symptoms):
        h     = self.net(symptoms)
        gamma = torch.sigmoid(self.gamma_head(h)) * 2.0
        beta  = self.beta_head(h)
        return gamma, beta

class _FiLMFusionModel(nn.Module):
    def __init__(self, feature_dim=1024, symptom_dim=36, num_classes=14):
        super().__init__()
        self.symptom_encoder = _SymptomEncoder(symptom_dim, feature_dim)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim), nn.Dropout(p=0.4),
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, image_features, symptoms):
        gamma, beta = self.symptom_encoder(symptoms)
        return self.classifier(gamma * image_features + beta)

# ── Model Auto-detection & Loading ───────────────────
def load_backbone():
    """Auto-detects and loads the image backbone."""
    # Priority 1: Confirmed safe model
    SAFE_MODEL = "densenet121_BEST_auc0.8031_ep19_SAFE.pth"
    if os.path.exists(SAFE_MODEL):
        print(f"Loading image model: {SAFE_MODEL}")
        m = models.densenet121(weights=None)
        m.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(in_f), nn.Dropout(p=0.4),
            nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(p=0.3),   nn.Linear(512, 14)
        )
        m.load_state_dict(torch.load(SAFE_MODEL, map_location=device))
        return m, 224, 1024, "DenseNet-121"

    # Priority 2: EfficientNet B4
    EFF_MODEL = "efficientnet_b4_14class_best.pth"
    if os.path.exists(EFF_MODEL):
        print(f"Loading image model: {EFF_MODEL}")
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
        return m, 380, 1792, "EfficientNet-B4"

    # Priority 3: Fallback DenseNet
    FALLBACK = "densenet121_14class_best.pth"
    if os.path.exists(FALLBACK):
        print(f"Loading image model: {FALLBACK}")
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
        return m, 224, 1024, "DenseNet-121"

    print("❌ No valid image model weights found! Place densenet121_BEST_auc0.8031_ep19_SAFE.pth in root.")
    exit()

backbone, img_size, feature_dim, model_name = load_backbone()
backbone.eval()
backbone.to(device)

# Load FiLM model weights
film_pattern = "film_efficientnet_best_auc*.pth" if "efficientnet" in model_name.lower() else "film_fusion_best_auc*.pth"
film_paths = sorted(glob.glob(film_pattern), reverse=True)
if not film_paths:
    print(f"❌ No FiLM checkpoints found for pattern {film_pattern}!")
    exit()

film_path = film_paths[0]
print(f"Loading FiLM model: {film_path}")
film_model = _FiLMFusionModel(feature_dim=feature_dim).to(device)
film_model.load_state_dict(torch.load(film_path, map_location=device))
film_model.eval()

# ── Dataset Loading ──────────────────────────────────
print("\nStep 1 — Loading labels and test split...")
if not os.path.exists("nih_full_labels.csv") or not os.path.exists("test_list.txt"):
    print("❌ Missing labels or test split files in workspace!")
    exit()

df_labels = pd.read_csv("nih_full_labels.csv")
with open("test_list.txt", "r") as f:
    test_filenames = {line.strip() for line in f}

test_df = df_labels[df_labels["Image Index"].isin(test_filenames)].reset_index(drop=True)
print(f"Locked test records: {len(test_df):,}")

# Resolve absolute image paths
def find_image_path(filename):
    for folder in os.listdir("."):
        if folder.startswith("images_"):
            path = os.path.join(folder, "images", filename)
            if os.path.exists(path):
                return path
    return None

test_df["resolved_path"] = test_df["Image Index"].apply(find_image_path)
test_df = test_df[test_df["resolved_path"].notna()].reset_index(drop=True)
print(f"Resolved paths: {len(test_df):,}")

# ── Dataset Definition ────────────────────────────────
class XRayFeatureDataset(Dataset):
    def __init__(self, df, img_size):
        self.df = df
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["resolved_path"]).convert("L")
        tensor = self.transform(img)
        labels = torch.tensor(row[DISEASES].values.astype(np.float32))
        return tensor, labels, idx

test_loader = DataLoader(test_df, batch_size=64, shuffle=False, num_workers=0)
dataset = XRayFeatureDataset(test_df, img_size)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

# ── Feature Extraction ────────────────────────────────
print(f"\nStep 2 — Extracting {feature_dim}-dim features from test images...")
extracted_features = []
ground_truth_labels = []

with torch.no_grad():
    for tensors, labels, _ in tqdm(loader, desc="Extracting features"):
        tensors = tensors.to(device)
        feat_map = backbone.features(tensors)
        if "densenet" in model_name.lower():
            feat_map = F.relu(feat_map, inplace=False)
        pooled = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(feat_map.size(0), -1)
        extracted_features.append(pooled.cpu().numpy())
        ground_truth_labels.append(labels.numpy())

features_arr = np.vstack(extracted_features)
labels_arr = np.vstack(ground_truth_labels)
print(f"Features shape: {features_arr.shape}  Labels shape: {labels_arr.shape}")

# ── EHR Symptom Vector Generator ──────────────────────
def sample_symptom_vector(true_diseases, rng):
    """Generates a symptom vector based on active diseases."""
    vec = np.zeros(SYMPTOM_DIM, dtype=np.float32)
    if not true_diseases:
        return vec
    for s_idx, symptom in enumerate(SYMPTOM_COLS):
        max_prev = 0.0
        for disease in true_diseases:
            if disease in SYMPTOM_PREVALENCE_BY_DISEASE:
                prev = SYMPTOM_PREVALENCE_BY_DISEASE[disease].get(symptom, 0.0)
                max_prev = max(max_prev, prev)
        vec[s_idx] = 1.0 if rng.random() < max_prev else 0.0
    return vec

# ── Generating Symptom Vectors ───────────────────────
print("\nStep 3 — Simulating clinical history cohorts...")
rng = random.Random(42)

# Cohort A: Coherent
coherent_symptoms = []
for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    pos_d = [d for d in DISEASES if row[d] == 1]
    coherent_symptoms.append(sample_symptom_vector(pos_d, rng))
coherent_symptoms = np.vstack(coherent_symptoms)

# Helper to add noise to symptoms
def add_noise(symptom_matrix, flip_rate, rng):
    matrix_noisy = symptom_matrix.copy()
    num_elements = matrix_noisy.size
    flip_indices = rng.sample(range(num_elements), int(num_elements * flip_rate))
    flat = matrix_noisy.ravel()
    for idx in flip_indices:
        flat[idx] = 1.0 - flat[idx]
    return matrix_noisy

# Cohort B: Noisy (10%, 20%, 30%)
noisy_10_symptoms = add_noise(coherent_symptoms, 0.10, rng)
noisy_20_symptoms = add_noise(coherent_symptoms, 0.20, rng)
noisy_30_symptoms = add_noise(coherent_symptoms, 0.30, rng)

# Cohort C: Contradictory (Generate symptoms for a different disease set)
contradictory_symptoms = []
for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    pos_d = [d for d in DISEASES if row[d] == 1]
    if pos_d:
        # Pick a disease they DO NOT have and generate symptoms for it
        neg_d = list(set(DISEASES) - set(pos_d))
        contra_d = [rng.choice(neg_d)]
    else:
        # Healthy, generate symptoms for a random severe disease
        contra_d = [rng.choice(DISEASES)]
    contradictory_symptoms.append(sample_symptom_vector(contra_d, rng))
contradictory_symptoms = np.vstack(contradictory_symptoms)

# Baseline placeholder: Uniform 0.3
placeholder_symptoms = np.ones((len(test_df), SYMPTOM_DIM), dtype=np.float32) * 0.3

# ── Run FiLM Inference for each Cohort ────────────────
def run_film_eval(symptom_matrix, name):
    print(f"Running evaluation for cohort: {name}")
    predictions = []
    
    # Batch processing through FiLM MLP
    batch_size = 256
    num_batches = int(np.ceil(len(features_arr) / batch_size))
    
    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min(len(features_arr), start + batch_size)
            
            feat_batch = torch.tensor(features_arr[start:end]).to(device)
            symp_batch = torch.tensor(symptom_matrix[start:end]).to(device)
            
            logits = film_model(feat_batch, symp_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.append(probs)
            
    preds_arr = np.vstack(predictions)
    
    # Calculate AUCs
    aucs = {}
    for d_idx, disease in enumerate(DISEASES):
        y_true = labels_arr[:, d_idx]
        y_pred = preds_arr[:, d_idx]
        if y_true.sum() > 0:
            aucs[disease] = roc_auc_score(y_true, y_pred)
        else:
            aucs[disease] = np.nan
    return aucs

cohorts = {
    "Baseline (Uniform 0.3)": placeholder_symptoms,
    "Contradictory History" : contradictory_symptoms,
    "Noisy (30% Noise)"     : noisy_30_symptoms,
    "Noisy (20% Noise)"     : noisy_20_symptoms,
    "Noisy (10% Noise)"     : noisy_10_symptoms,
    "Coherent Clinical History": coherent_symptoms,
}

results = {}
for name, matrix in cohorts.items():
    results[name] = run_film_eval(matrix, name)

# ── Print Results Table ───────────────────────────────
print("\n" + "=" * 65)
print("              SUMMARY OF FiLM ABLATION STUDY RESULTS")
print("=" * 65)

# Format markdown table output
header = f"| Disease | { ' | '.join(cohorts.keys())} |"
separator = f"|---| { ' | '.join(['---'] * len(cohorts))} |"
print(header)
print(separator)

table_lines = [header, separator]

for disease in DISEASES:
    line = f"| {disease} |"
    for name in cohorts.keys():
        val = results[name][disease]
        line += f" {val:.4f} |" if not np.isnan(val) else " NaN |"
    print(line)
    table_lines.append(line)

# Mean AUC Row
mean_line = "| **MEAN AUC** |"
for name in cohorts.keys():
    vals = [results[name][d] for d in DISEASES if not np.isnan(results[name][d])]
    mean_line += f" **{np.mean(vals):.4f}** |"
print(mean_line)
table_lines.append(mean_line)
print("=" * 65)

# Save results to markdown
out_file = "film_simulated_test_results.md"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("# FiLM Simulated EHR Evaluation Study Results\n\n")
    f.write("\n".join(table_lines))
    f.write("\n")
print(f"✅ Results compiled and saved to: {out_file}")
