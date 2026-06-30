# AyShCXR — Chest X-ray Multi-Disease Classifier

A multi-architecture deep-learning system for classifying 14 thoracic diseases on
chest X-rays, optimized for deployment in rural Indian Primary Health Centers (PHCs).

**Authors:** Subhrakant Sethi, Ayush Singh
**Institution:** Thapar Institute of Engineering & Technology (B.Tech CSE, 2nd Year)
**Status:** Research prototype — not for clinical use.

## Performance

Rather than relying on a single model, we benchmarked and ensembled three distinct
architectures on the CheXpert dataset.

| Model                          | Mean AUC | Notes                                  |
|--------------------------------|----------|----------------------------------------|
| Radiologist (Wang et al. 2017) | 0.778    | Human benchmark                        |
| Rad-DINO (ours)                | 0.8208   | CheXpert test set; best in 3-fold CV   |
| EfficientNet-B4 (ours)         | 0.8199   | CheXpert test set                      |
| DenseNet-121 (ours)            | 0.8273   | CheXpert test set                      |
| **Ensemble (3-model)**         | **0.8377** | CheXpert test set — headline result  |
| Ensemble → NIH (cross-dataset) | 0.8089   | NIH official test, 7 shared diseases   |

**Key finding:** three very different architectures cluster within 0.006 AUC,
indicating the task is **label-noise-limited** (F1 ceiling ≈ 0.44), not
model-capacity-limited. Cross-dataset transfer to NIH stays within ~0.03 of
in-domain performance — strong generalization across hospitals.

## Architecture

Three backbones trained under an identical patient-level split and loss recipe:

- **Rad-DINO** @ 224×224, 87M params — Microsoft medical-image foundation model
- **DenseNet-121** @ 320×320, 7.5M params
- **EfficientNet-B4** @ 380×380, 19M params
- **Loss:** Focal Loss (γ=2.0, α=0.75) + Label Smoothing (0.1) + per-disease weights
- **Optimizer:** Layer-wise LR Decay AdamW
- **Augmentation:** Mixup (β=0.4), RandAugment, GaussianBlur (rural scanner simulation)
- **Training:** AMP + gradient accumulation, `torch.compile`
- **No horizontal flip** — clinically invalid for CXR (heart asymmetry / dextrocardia)

## Disease loss weights

Normalized from positive counts, with clinical priority overrides:
- Pneumonia weight = 2.00 (highest mortality in rural India)
- Hernia weight = 3.00 (rarest disease)

## Two-Stage Inference Pipeline

- **Stage 1 — Image:** backbone + MC Dropout (20 passes) for uncertainty quantification
- **Stage 2 — Symptoms:** per-disease targeted clinical question bank
- **Fusion:** 0.9 × image score + 0.1 × symptom evidence
- **Output:** GradCAM heatmap + timestamped clinical report

## Repository Structure

```
├── chexpert_raddino_train.py       # Rad-DINO training (CheXpert)
├── chexpert_densenet_train.py      # DenseNet-121 training (CheXpert)
├── chexpert_efficientnet_train.py  # EfficientNet-B4 training (CheXpert)
├── ensemble_eval.py                # 3-model ensemble evaluation
├── eval_full_metrics.py            # Acc/Prec/Rec/F1/Sens/Spec + AUC table
├── cross_dataset_eval.py           # CheXpert → NIH generalization test
├── cv_chexpert.py                  # 3-fold cross-validation
├── save_chexpert_preds.py          # Save test predictions
├── nih_efficientnet_train.py       # NIH ChestX-ray14 training
├── build_chexpert_table.py         # Metrics table builder
├── predict_single.py               # Single image inference + MC Dropout + GradCAM
├── gradcam_visualize.py            # GradCAM heatmap generation
├── stage2_questions.py             # Symptom questionnaire bank
├── app.py                          # Streamlit demo UI
└── requirements.txt                # Exact dependency versions
```

## Setup

```bash
git clone https://github.com/AyShCXR/ayshcxr-model.git
cd ayshcxr-model
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
# Install PyTorch with CUDA 12.4 support first
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Then install remaining dependencies
pip install -r requirements.txt
```

**Note (Rad-DINO):** loading `microsoft/rad-dino` requires
`transformers>=4.48,<5` and `huggingface_hub<1.0` — install with `--no-deps`
to avoid disturbing your PyTorch build.

## Dataset

Trained on **CheXpert** and evaluated cross-dataset on **NIH ChestX-ray14**:
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- NIH images: https://nihcc.app.box.com/v/ChestXray-NIHCC
- NIH labels CSV: https://www.kaggle.com/datasets/nih-chest-xrays/data

Both datasets are research-licensed and are **not** redistributed in this repo.

## Model weights

Pretrained `.pth` files are not in this repo (file size + dataset licensing).
[TODO: add HuggingFace Hub link once uploaded]

## Usage

Train a CheXpert backbone:
```bash
python chexpert_raddino_train.py        # or chexpert_densenet_train.py / chexpert_efficientnet_train.py
```

Evaluate the ensemble:
```bash
python ensemble_eval.py
```

Cross-dataset (CheXpert → NIH):
```bash
python cross_dataset_eval.py
```

Single image inference:
```bash
python predict_single.py --image path/to/xray.png
```

Web UI:
```bash
streamlit run app.py
```

## Hardware

| Stage                   | Hardware                                         |
|-------------------------|--------------------------------------------------|
| Development / Inference | HP Victus 15, i5-13420H, RTX 4050 6GB, 16GB RAM  |
| Full training           | University GPU server — NVIDIA H100 (MIG slice)  |

## Roadmap

- [ ] Upload pretrained weights to HuggingFace Hub
- [ ] Real-symptom (non-leaking) multimodal fusion with patient history
- [ ] Per-disease threshold calibration for deployment
- [ ] 3-phase anatomical pretraining (ImageNet → Montgomery/Shenzhen → CheXpert)

## Known Issues / Design Decisions

- **No horizontal flip augmentation** — heart is left-sided; flipping creates an artificial dextrocardia pattern
- **EfficientNet-B4 at 224px underperforms** — use 380px (native resolution)
- **Asymmetric Loss (ASL) tested and dropped** — caused Pneumonia AUC regression
- **Label noise is the ceiling** — report-level labels stall training; impression-level labels are required to learn

## Disclaimer

This is a research prototype. It is not an FDA/CDSCO approved medical device.
Do not use for actual patient diagnosis. All predictions must be verified by a
qualified radiologist.

## License

Private repository — all rights reserved. Contact authors for collaboration.

## Citation

If this work informs your research:

> Sethi, S., & Singh, A. (2026). AyShCXR: Multi-disease chest X-ray classification
> for rural PHC deployment. Thapar Institute of Engineering & Technology.
