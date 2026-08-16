# AyShCXR — Chest X-ray Multi-Disease Classifier

EfficientNet-B4 based classifier for 14 thoracic diseases on NIH ChestX-ray14,
optimized for deployment in rural Indian Primary Health Centers (PHCs).

**Authors:** Subhrakant Sethi, Ayush Singh
**Institution:** Thapar Institute of Engineering & Technology (B.Tech CSE, 2nd Year)
**Status:** Research prototype — not for clinical use.

---

## Performance

| Model | Mean AUC | Notes |
|---|---|---|
| Radiologist (Wang et al. 2017) | 0.778 | Human benchmark |
| DenseNet-121 (ours, baseline) | 0.8031 | Official NIH test set |
| DenseNet-121 (ours, val) | 0.8339 | Best validation AUC |
| EfficientNet-B4 (current) | TBD | Training in progress @ 380px |

---

## Architecture

- **Backbone:** EfficientNet-B4 @ 380×380, 19M parameters
- **Loss:** Focal Loss (γ=2.0, α=0.75) + Label Smoothing (0.1)
- **Optimizer:** Layer-wise LR Decay, 4 groups (1e-6 → 3e-5)
- **Augmentation:** Mixup (β=0.4), RandAugment, GaussianBlur (rural scanner simulation)
- **Training:** AMP, gradient accumulation (effective batch 128)
- **No horizontal flip** — clinically invalid for CXR (heart asymmetry / dextrocardia)

### Disease loss weights
Normalized from NIH positive counts, with clinical priority overrides:
- Pneumonia weight = 2.00 (highest mortality in rural India)
- Hernia weight = 3.00 (rarest disease)

---

## Two-Stage Inference Pipeline

1. **Stage 1 — Image:** EfficientNet-B4 + MC Dropout (20 passes) for uncertainty quantification
2. **Stage 2 — Symptoms:** Per-disease targeted clinical question bank
3. **Fusion:** 0.9 × image score + 0.1 × symptom evidence
4. **Output:** GradCAM heatmap + timestamped clinical report

---

## Repository Structure

```
├── build_and_train_demo.py    # Main training script (EfficientNet-B4, professor GPU)
├── build_and_train_laptop.py  # Laptop validation script (DenseNet-121, 224px)
├── predict_single.py          # Single image inference + MC Dropout + GradCAM
├── eval_and_save_preds.py     # Full test set evaluation (25,596 images)
├── gradcam_visualize.py       # GradCAM heatmap generation
├── stage2_questions.py        # Symptom questionnaire bank
├── check_results.py           # Training history analysis
├── app.py                     # Streamlit demo UI
└── requirements.txt           # Exact dependency versions
```

---

## Setup

```bash
git clone https://github.com/AyushSingh-Ww/ayshcxr-model.git
cd ayshcxr-model
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install PyTorch with CUDA 12.4 support first
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Then install remaining dependencies
pip install -r requirements.txt
```

### Dataset

Download NIH ChestX-ray14 (~45GB):
- Images: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Labels CSV: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Place images in `images_001/images/` through `images_012/images/`
- Run `python prepare_labels.py` to generate `nih_full_labels.csv`

### Model weights

Pretrained `.pth` files are not in this repo (file size limits).
**[TODO: add HuggingFace Hub or Google Drive link once uploaded]**

---

## Usage

### Train (professor GPU — EfficientNet-B4 @ 380px)
```bash
python build_and_train_demo.py
```

### Train (laptop — DenseNet-121 @ 224px for testing)
```bash
python build_and_train_laptop.py
```

Both scripts auto-resume from the last checkpoint if interrupted.

### Evaluate on NIH test set
```bash
python eval_and_save_preds.py
```

### Single image inference
```bash
python predict_single.py --image path/to/xray.png
```

### Web UI
```bash
streamlit run app.py
```

---

## Hardware

| Stage | Hardware |
|---|---|
| Development / Inference | HP Victus 15, i5-13420H, RTX 4050 6GB, 16GB RAM |
| Full training | University GPU server (professor) |

---

## Roadmap

- [ ] FFT parallel branch (frequency domain texture features)
- [ ] Pixel-level density quantification per anatomical zone
- [ ] 3-phase anatomical pretraining (ImageNet → Montgomery/Shenzhen → NIH)
- [ ] FiLM symptom injection into backbone
- [ ] CBAM zone attention (radiologist blind spot regions)

---

## Known Issues / Design Decisions

- No horizontal flip augmentation — heart is left-sided; flipping creates artificial dextrocardia pattern
- EfficientNet-B4 at 224px underperforms — use 380px (native resolution) for GPU runs
- Asymmetric Loss (ASL) was tested and dropped — caused Pneumonia AUC regression

---

## Disclaimer

This is a **research prototype**. It is **not** an FDA/CDSCO approved medical device.
Do not use for actual patient diagnosis. All predictions must be verified by a
qualified radiologist.

---

## License

Private repository — all rights reserved. Contact authors for collaboration.

---

## Citation

If this work informs your research:
```
Sethi, S., & Singh, A. (2026). AyShCXR: Multi-disease chest X-ray classification
for rural PHC deployment. Thapar Institute of Engineering & Technology.
```
