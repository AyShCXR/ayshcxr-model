# AyShCXR

**Multi-label chest X-ray screening for rural Primary Health Centres, with per-finding reliability validation.**

Subhrakant Sethi & Ayush Singh — B.Tech CSE, Thapar Institute of Engineering and Technology, Patiala

---

## What this is

India has roughly one radiologist per 100,000 people. Rural Primary Health Centres take chest X-rays but have nobody qualified to read them. AyShCXR is a decision-support system that reads the film on-site, combines it with the patient's reported symptoms and history, and produces a triage decision — refer today, review in two weeks, or routine care.

It runs entirely offline. No image leaves the device.

## What makes it different

Most chest X-ray systems report every finding with a confidence score. This one **measures whether it can actually detect each finding, and refuses to report the ones it cannot.**

Of 21 findings analysed, **14 met a validation bar of AUC ≥ 0.70 and are reported as diagnoses. Five are deliberately withheld**, with an explanation shown to the clinician:

> *"This X-ray shows a pattern that may indicate Pleural Thickening, but AyShCXR cannot assess it reliably. In our validation it performed no better than chance (AUC 0.50 on 146 tested cases), so we do not report it as a diagnosis. Please have a doctor review this film."*

## Results

**Models** — three architectures trained on ~300,000 radiographs (CheXpert Plus, NIH ChestX-ray14) with patient-level splits:

| Model | Test AUC |
|---|---|
| DenseNet-121 | 0.8281 |
| Rad-DINO (ViT) | 0.8238 |
| EfficientNet-B4 | 0.8226 |
| **Ensemble** | **0.838** |

**External validation** — zero-shot on NIH, never seen in training: **0.809 AUC** across the 7 shared findings.

**Per-finding detection**, measured on stratified NIH films:

| | AUC | | AUC |
|---|---|---|---|
| Emphysema | 0.911 | Effusion | 0.829 |
| Hernia | 0.911 | Mass | 0.812 |
| Cardiomegaly | 0.914 | Atelectasis | 0.791 |
| Pneumothorax | 0.896 | Nodule | 0.776 |
| Edema | 0.883 | Pneumonia | 0.745 |
| | | **Pleural Thickening** | **0.497 — withheld** |

**Edge deployment** — DenseNet-121 exported to ONNX and INT8-quantised:

| | Size | Latency (4 CPU threads, no GPU) |
|---|---|---|
| PyTorch | 29.2 MB | 210 ms |
| ONNX FP32 | 28.9 MB | 101 ms |
| ONNX INT8 | **8.0 MB** | 192 ms |

Quantisation shifts probabilities by 0.033 on average; the top finding changes on 20% of images, and 83% of those had an original top-1/top-2 margin below 0.05 — near-ties, not confident predictions being overturned.

## Safety behaviours

- **Input validation** — rejects images that are not frontal chest radiographs (checks size, colour, contrast, and the bright-spine/dark-lung signature)
- **Quality assessment** — flags rotation, over/under-exposure and motion blur, and names which findings are unreliable on that specific film. Calibrated on 400 real radiographs; 74.5% pass cleanly.
- **Calibrated probabilities** — per-disease temperature scaling reduced expected calibration error by 88.7%. A displayed "70%" means 70%.
- **Abstention** — the system declines when its models disagree. Disagreement between models is measured explicitly, not discarded.
- **Non-pathologies excluded** — "Support Devices" and "No Finding" can never be reported as a diagnosis.

## Layout

```
core/          the running system (13 files)
training/      model training scripts
evaluation/    AUC, cross-validation, cross-dataset evaluation
reports/       verification, threshold tuning, ONNX export
data_prep/     dataset preparation
results/       calibration, thresholds, baselines (required at runtime)
docs/          project reports
run_tests.py   27-test suite
```

## Running it

Requires Python 3.13, and model checkpoints (not in this repo — see below).

```bash
python -m venv .venv-app
.venv-app/Scripts/activate
pip install torch torchvision flask opencv-python pandas scikit-learn onnxruntime
python core/app.py
```

Then open http://127.0.0.1:5000

Command line:

```bash
python core/predict_single.py xray.png --symptoms fever_high,cough_productive
```

Tests:

```bash
python run_tests.py
```

## Not in this repository

**Model weights** (~923 MB) — trained on CheXpert Plus and NIH ChestX-ray14, both research-use datasets. Not redistributable here.

**Datasets** (~44 GB) — download from the original sources: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC), [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), [VinDr-CXR](https://physionet.org/content/vindr-cxr/).

## Honest limitations

- **Not a diagnostic device.** A research prototype for clinical assistance, requiring review by a qualified professional.
- **Not state of the art.** Published CheXpert leaderboard results (~0.90+) are on a 5-disease subset scored against radiologist consensus. This is all-14, scored against NLP-generated labels, on a patient-level split — a harder and non-identical setting.
- **Performance is label-limited, not capacity-limited.** Three architecturally distinct models converged within 0.006 AUC because the training labels agree with radiologists at only F1 ≈ 0.44. This is the project's central finding.
- **Naming the correct finding is harder than detecting one.** On films with a single confirmed finding, the correct one is named first 43% of the time (random would be 7%), ranging from 84% (Hernia) to 2% (Infiltration).
- **Healthy films.** 59% of confirmed-normal radiographs are correctly reported as normal. The remainder receive at least one false finding.
- **Edge deployment is benchmarked on x86 only.** INT8 measured slower than FP32 on our test machine — dynamic quantisation overhead, and no VNNI acceleration. ARM targets are expected to differ but have not been measured, and no ARM figure is claimed.
- Trained on research datasets from the USA. Deployment in India would require validation on locally collected data and regulatory approval.

## Licence

Code: MIT. Datasets and model weights are governed by their own licences and are not included.
