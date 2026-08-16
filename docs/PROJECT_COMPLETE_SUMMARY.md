# AyShCXR — Complete Project Summary (Day 1 → Now)
### Exhaustive record of everything done, every model, every fix, every decision

**Project:** AyShCXR — AI chest X-ray (CXR) disease screening for rural Indian Primary Health Centres (PHCs), where no radiologist is available.
**Team:** Subhrakant Sethi & Ayush Singh — B.Tech 2nd year, Thapar Institute of Engineering and Technology (TIET), Patiala.
**Working directory:** `C:\AyShCXR` (laptop, Windows 11, RTX 4050 6GB).
**GPU server:** `dgxhnode5` — NVIDIA H100 80GB, MIG 3g.40gb slice (~40GB), accessed via JupyterLab. (Earlier server was at `172.16.224.121:1008`; current at `172.16.224.125:1020`.)

---

# PART 0 — THE BIG PICTURE

The project has gone through **three major phases**:

1. **Phase 1 (the original work, ~April–May 2026):** NIH + CheXpert v1.0 EfficientNet models + FiLM symptom fusion → first Word report submitted to the professor.
2. **Phase 2 (mid-June 2026):** Moved to the newer **CheXpert Plus** dataset; hit and solved a 2-day "label-noise" wall where validation AUC was stuck at 0.50.
3. **Phase 3 (June 19–22, 2026 — this session):** Built a rigorous **multi-architecture comparison** (EfficientNet vs DenseNet vs Rad-DINO transformer) on CheXpert Plus, an **ensemble**, the full **professor-format metrics table**, **3-fold cross-validation**, **cross-dataset evaluation**, and a complete **Word report**.

---

# PART 1 — PHASE 1: THE ORIGINAL MODELS (April–May 2026)

This work predates this chat but is the foundation. It produced the first report (`AyShCXR_Project_Report_v2.docx`).

## 1.1 Models trained (Phase 1)
| Model | Trained on | Result |
|---|---|---|
| NIH EfficientNet-B4 | NIH ChestX-ray14 (86,524 train) | **Test AUC 0.7647** (locked test 25,596; val 0.8009 @ ep60) |
| CheXpert v1.0 EfficientNet-B4 | CheXpert v1.0 (224,316) | **Cross-dataset CheXpert→NIH 0.6860** (val 0.8072 @ ep52) |
| FiLM + Symptoms (NIH base) | NIH + 36-dim symptom vectors | **val 0.9085 / test 0.6878** |
| Radiologist benchmark | — | 0.778 |

- NIH model was only **1.3% below the radiologist benchmark** (0.7647 vs 0.778).
- Cross-dataset (CheXpert→NIH): shared diseases scored 0.74–0.87; the 4 diseases absent from CheXpert scored 0.40–0.56 — clean evidence of dataset shift.

## 1.2 The FiLM symptom fusion — and its hidden flaw
- Architecture: 36-dim symptom vector → SymptomEncoder → γ (gamma) + β (beta) vectors → modulates the 1792-dim image features (`modulated = γ × features + β`). Frozen backbone, 3-phase training (cache features → train FiLM → eval).
- **Validation AUC was 0.9085** but **test AUC only 0.6878**.
- **CRITICAL (re-confirmed this session):** the 0.9085 used **simulated symptom vectors generated FROM the disease labels** (`SYMPTOM_PREVALENCE_BY_DISEASE` in `medical_knowledge.py`). That is **label leakage** — the symptoms encoded the answer. The honest test (uniform 0.3 placeholder symptoms) gave 0.6878. **The FiLM 0.95–0.97 numbers must NEVER be presented as a model AUC.** A legitimate version would use real, independent inputs (demographics + the report's History section), not label-derived symptoms.

## 1.3 Phase-1 architecture (both image models, identical)
- EfficientNet-B4, **380px, 1-channel grayscale**, ~19M params, 1792-dim features.
- Head: BN → Linear → Linear (14 classes).
- Loss: Focal (γ=2.0, α=0.75) + label smoothing 0.1.
- Optimizer: 4-group LLRD AdamW (1e-6 / 5e-6 / 1e-5 / 3e-5).
- Batch 128 (32 × grad-accum 4), AMP.
- Augmentation: Mixup β=0.4, GaussianBlur, RandAugment, RandomErasing; **NO horizontal flip** (the heart is left-sided — flipping creates a false dextrocardia pattern).
- Split: MultilabelStratifiedShuffleSplit (image-level) on the official NIH train_val pool; official `test_list.txt` = locked test.

## 1.4 Phase-1 bugs fixed (historical record)
- `torch.compile` adding `_orig_mod.` prefix to checkpoint keys.
- DataLoader workers bottleneck: 29 min/epoch → 8 min/epoch.
- CheXpert zip extracted to batch-named folders → built a path index.
- Soft uncertain labels (−1 → 0.5) silently breaking AUC → fixed by filtering to definite labels.
- BatchNorm crash on a single-sample last batch → `drop_last=True`.
- AUC showing 0.0000 for 30 CheXpert epochs before the soft-label fix.
- FiLM eval: torchvision/timm mismatch, 1-channel fix, images spread across `images_001/`–`images_006/`, wrong normalization `[0.5],[0.5]` → correct `[0.485],[0.229]`.

## 1.5 Phase-1 things that FAILED (do not retry)
- Asymmetric Loss (ASL) γ_neg=4 → Pneumonia AUC dropped.
- Adding OLD CheXpert (2019) → 0.7830 (noisy −1 labels, bad mapping).
- Progressive resizing + curriculum together → timing collision.
- EfficientNet-B4 at 224px → wrong resolution, underperformed.
- Horizontal flip in TTA → clinically invalid.
- AUC-maximization with PESG optimizer → breaks LLRD.

---

# PART 2 — PHASE 2: CheXpert Plus & THE 2-DAY LABEL-NOISE WALL

## 2.1 The problem
On the newer **CheXpert Plus** dataset (~190,800 frontal images, real radiology reports, CheXBERT-generated labels), every training run had the model **fit the training data but get validation AUC stuck at exactly 0.50 (random)** — even for trivially visible classes like Support Devices.

## 2.2 What was ruled out (exhaustively, do not re-investigate)
- **Images:** verified via a montage — clean, distinct X-rays; resize to 412px faithful.
- **Pipeline:** overfit test memorized 200 images to AUC 1.0 → model/optimizer/pipeline all work.
- **BatchNorm eval mode:** ruled out with a 3-mode AUC check (eval / train-BN / fixed-BN all 0.50).
- **More epochs:** ran 23 epochs, train loss 0.125, val still 0.50.
- **torch.compile:** wrongly blamed; it actually works fine.
- Focal vs BCE, learning rates, head complexity, 1 vs 3 channels, pos_weight, grad clipping, patient vs image split — none fixed it.

## 2.3 THE FIX (root cause)
The labels came from **`report_fixed.json`** = CheXBERT run on the **ENTIRE radiology report** (including clinical history and prior-study comparisons). Those tagged pathologies **not visible in the current frontal image**, creating label noise the model couldn't learn from (so it memorized instead).

**Fix: switch to `impression_fixed.json`** = CheXBERT on just the radiologist's **Impression** (their conclusion about the current study). Clean enough to learn. **Val AUC jumped 0.50 → 0.756 at epoch 1**, then 0.783, climbing.
- ⚠️ `findings_fixed.json` is **degenerate** ("No Finding = 1" everywhere) — never use it.
- This is a citable negative result on label provenance.

## 2.4 The first working CheXpert Plus model
- `chexpert_train.py`: EfficientNet-B4, **3-channel** (RGB, ImageNet norm), complex head (BN→Linear512→GELU→Linear14), LLRD AdamW, Focal loss, patient-level split (GroupShuffleSplit, overlap 0), torch.compile, BN-recompute eval.
- **Peaked at epoch 11 = 0.8164**, then overfit (declined to ~0.80 by epoch 35). Best checkpoint saved at `chexpert_best_auc0.8164_ep11.pth`.

---

# PART 3 — PHASE 3 (THIS SESSION): MULTI-MODEL COMPARISON

## 3.1 The goal shift
Instead of one model, the user wanted a **professor-format comparison across architectures** (like the professor's BME-lecture template — multiple models, a metrics table, AUC).

## 3.2 The unified harness (`train_model.py`)
One script, swap `--arch`, identical everything else (patient split seed 42, Focal loss, 18 epochs, BN-fix eval). Supported EfficientNet-B4, DenseNet-121, Rad-DINO, Swin-T, ViT-B/16. This made a **fair comparison** possible.

## 3.3 The Rad-DINO install saga (the other big blocker)
Rad-DINO (`microsoft/rad-dino`, a medical vision transformer pretrained on 838K CXRs) wouldn't load due to a HuggingFace ↔ NVIDIA-PyTorch conflict. **Root cause (finally pinned down):**
- The nv torch is tagged **`2.4.0a0`** — an *alpha* prerelease, so `2.4.0a0 < 2.4.0` by PEP440. `transformers 5.x` requires `torch >= 2.4` → rejects the alpha → disables the PyTorch backend.
- `transformers 4.x` accepts the alpha torch but requires `huggingface_hub < 1.0`; the container had `hub 1.19.0` (too new) and a **corrupted `transformers/utils/versions.py`** (a prior manual edit to bypass a version check broke its indentation).

**THE FIX (verified, never touch torch):**
- `transformers==4.57.6` (`pip install --force-reinstall --no-deps "transformers>=4.48,<5"`)
- `huggingface_hub==0.36.2` (`pip install --force-reinstall --no-deps "huggingface_hub>=0.34,<1.0"`)
- `tokenizers==0.22.2` (left as-is).
- Confirmed with `is_torch_available() == True`; Rad-DINO loaded (hidden=768). Saved in memory as `raddino_install_fix`.

## 3.4 First CheXpert comparison (train_model.py)
| Model | Val AUC (all-14) |
|---|---|
| DenseNet-121 (320px) | 0.8235 |
| Rad-DINO (224px) | 0.8193 |
| EfficientNet-B4 (380px) | 0.8164–0.8178 |
| **Ensemble (avg of 3)** | **0.8367** |

**Key finding:** three very different architectures cluster within 0.006 AUC → performance is **label-noise-limited (CheXBERT F1≈0.44 ceiling), not capacity-limited.**

## 3.5 Research-paper analysis
The user shared a 9-paper research synthesis PDF. The highest-ROI, low-risk improvement identified: **GMP+GAP dual pooling** (Paper 4 — ablation-proven ~+1% on NIH; GAP captures diffuse findings, GMP captures focal ones like nodules/pneumothorax). Other techniques (CLIP/contrastive/metric losses) deferred as too expensive for the deadline. The **label-ceiling F1=0.44** finding (Papers 2 & 6) became the central explanatory citation.

---

# PART 4 — PHASE 4 (THIS SESSION): THE NIH DETOUR (later dropped)

## 4.1 What happened
The user wanted NIH results too (believing NIH could hit 0.85+). Downloaded NIH ChestX-ray14 (42GB) via **kagglehub** to `/workspace/kagglehub_cache/.../versions/3`. Discovered the user already had `nih_full_labels.csv` + `train_val_list.txt` + `test_list.txt` locally (only the image paths needed re-pointing → `fix_nih_paths.py`).

## 4.2 NIH EfficientNet training — and the bugs fixed
Ran `nih_efficientnet_train.py` (a copy of the proven `build_and_train_demo.py`, then enhanced with 3-channel + GMP+GAP + RandomPerspective). Bugs hit and fixed, in order:
1. **1-channel sanity assert** still hardcoded after the 3-channel switch → fixed to expect 3 channels.
2. **`/dev/shm` bus error** (crashed at validation): root cause = **`persistent_workers=True` on BOTH loaders** kept 4 train + 4 val workers alive, accumulating shared memory until validation. Fix: `val num_workers=0` + `train persistent_workers=False` + file-system sharing strategy.
3. **Slow training (~8–9 min/epoch):** NIH images are raw 1024px (heavy decode). `resize_nih.py` pre-resized to 412px → ~3× faster.
4. The conservative proven-pipeline LRs (head 3e-5) made it crawl (60 epochs). Switched to the fast recipe (head 2e-4, 18 epochs).

**Result: NIH EfficientNet finished at 0.7853 val** (above the 0.778 radiologist benchmark by +0.007). Honest conclusion: **NIH all-14 single models top out ~0.78–0.82** — 0.85+ is not achievable for an image model (the radiologist benchmark itself is 0.778).

## 4.3 NIH dropped
The user decided to **focus only on CheXpert** (NIH was slow, label-limited, and not going to hit 0.85). NIH was set aside.

---

# PART 5 — PHASE 5 (THIS SESSION): THE REBUILD WITH GMP+GAP

## 5.1 The "blunder" realization
The user noticed `train_model.py` had diverged from the **proven `build_and_train_demo.py`** pipeline that underpinned the authenticity/cross-dataset claims. After much discussion, the decision: build **3 dedicated CheXpert files**, one per architecture, sharing one consistent recipe.

## 5.2 The 3 CheXpert training files
`chexpert_efficientnet_train.py`, `chexpert_densenet_train.py`, `chexpert_raddino_train.py` — identical except the `ARCH` line. The recipe:
- **3-channel** RGB, ImageNet norm.
- **GMP+GAP dual pooling** for the CNNs (via a `CNNDualPool` wrapper); Rad-DINO uses its CLS token.
- Shared complex head: `BN → Dropout(0.4) → Linear(→512) → GELU → Dropout(0.3) → Linear(→14)`.
- **Focal Loss** (γ2, α0.75, smoothing 0.1) + per-disease weights.
- **2-group LLRD AdamW**: backbone 6e-5 (CNN)/3e-5 (transformer), head 2e-4.
- Cosine annealing, **18 epochs**, AMP, gradient clipping, torch.compile.
- Augmentation: resize→crop, rotation, affine, perspective, colour jitter, Gaussian blur (no h-flip).
- **shm-safe:** val `num_workers=0`, train non-persistent, file-system sharing.
- Reads `chexpert_clean.csv` (resized 412px).

## 5.3 Overnight chained run + results
Ran chained (Rad-DINO → EfficientNet → DenseNet). **All three beat their old versions:**
| Model | New best val AUC | (old) |
|---|---|---|
| DenseNet-121 | **0.8281** | 0.8235 |
| Rad-DINO | **0.8238** | 0.8193 |
| EfficientNet-B4 | **0.8226** | 0.8178 |

GMP+GAP genuinely helped the CNNs (~+0.5%).

## 5.4 Ensemble
`ensemble_chexpert.py` (matched to the GMP+GAP architecture): averaged the 3 models' probabilities.
**Ensemble = 0.8401 (val) / 0.8377 (proper held-out test).** All loaded `missing=0 unexpected=0` (architecture match confirmed).

---

# PART 6 — PHASE 6 (THIS SESSION): EVALUATION & DELIVERABLES

## 6.1 The proper train/val/test split
`save_chexpert_preds.py`: rebuilt the patient split as **Train (~175k) / Val (7,856) / Test (7,600)** — held-out 8% halved by patient into val + test. Saved all 3 models' predictions + labels to `chexpert_preds.npz` (3MB). This decoupled GPU work (inference) from laptop work (table building).

## 6.2 The metrics table (`build_chexpert_table.py`, laptop)
The professor's BME template was for **single-label tabular** data; the CXR task is **multi-label**. The script computes everything correctly:
- **Per-disease metrics, macro-averaged** (not single-label accuracy).
- **AUC = macro per-disease ROC-AUC**.
- Threshold-based metrics (Acc/Prec/Recall/F1/Sens/Spec) at a **per-disease threshold tuned on validation**.
  - First tried **Youden's J** → precision collapsed (~0.32) because it maximizes sensitivity at the cost of false positives.
  - Switched to **max-F1 threshold** → balanced: Acc/Spec ~0.86, Prec ~0.42, Recall/Sens ~0.55, F1 ~0.47.
- Added an **AUC (5-dx)** column (5 competition diseases) — which came out ~0.83, **not higher** than all-14 (because the 5 include hard Atelectasis/Consolidation while all-14 includes easy Pneumothorax/No Finding).
- Outputs: `CheXpert_Model_Comparison.csv`, `roc_curves.png`, `auc_bar.png`. Colab version: `colab_chexpert_table.py`.

## 6.3 Cross-validation (`cv_chexpert.py`) — why 3-fold not 5-fold
- True k-fold (fresh model retrained per fold, **GroupKFold** patient-grouped).
- **Full 5-fold of all 3 models = 15 retrainings ≈ 35 GPU-hours** → infeasible.
- Used a genuine but tractable config: **3 folds, 10 epochs/fold, 90k subset** (9 retrainings, ~3.5 GPU-hr).
- **Bug fixed:** the first version validated every epoch (slow with `num_workers=0`) — changed to **validate once per fold**, which made the timing fit.
- Results (extremely stable):
| Model | 3-fold CV AUC | std |
|---|---|---|
| Rad-DINO | **0.8089** | ±0.0003 |
| DenseNet-121 | 0.7983 | ±0.0010 |
| EfficientNet-B4 | 0.7644 | ±0.0013 |
- **Notable:** Rad-DINO leads the CV (even though DenseNet leads the full test) → under reduced training, its 838K-CXR medical pretraining makes it the most **data-efficient**.
- CV AUC reads lower than test AUC because the CV models are deliberately under-trained (10 ep/90k); its purpose is **stability**, not the headline number.

## 6.4 Cross-dataset evaluation (`cross_dataset_eval.py`)
CheXpert-trained models run **zero-shot on the NIH official test set** (25,596 images), scored on the **7 diseases common to both** (Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion=Effusion); the 4 NIH-only diseases (Emphysema, Fibrosis, Pleural Thickening, Hernia) excluded.
| Model | CheXpert→NIH AUC (7 common) |
|---|---|
| EfficientNet-B4 | 0.7907 |
| DenseNet-121 | 0.7990 |
| Rad-DINO | 0.7916 |
| **Ensemble** | **0.8089** |
- Per-disease (ensemble): Cardiomegaly **0.90**, Pneumothorax **0.90** (transfer near-perfectly); Consolidation 0.74, Pneumonia 0.73 (subtle, transfer less).
- **Only ~0.03 below in-domain (0.838 → 0.809)** = strong generalization → the model learned real radiographic features, not dataset artifacts. Cleaner than the old report's 0.6860 (which averaged in the absent diseases).

## 6.5 The macro vs micro AUC clarification
- The **table's AUC-ROC (0.838)** is **macro** (per-disease averaged — the honest headline).
- The **ROC-curve plot's AUC (~0.92)** is **micro** (all predictions pooled — inflated by easy negatives). **Do not headline 0.92.**

---

# PART 7 — THE FINAL NUMBERS (all in one place)

## In-domain CheXpert (held-out test, macro)
| Model | Test AUC | Test Acc | Test F1 | 5-dx AUC | 3-fold CV AUC |
|---|---|---|---|---|---|
| EfficientNet-B4 | 0.820 | 0.860 | 0.457 | 0.818 | 0.764 ±0.001 |
| DenseNet-121 | 0.827 | 0.859 | 0.466 | 0.820 | 0.798 ±0.001 |
| Rad-DINO | 0.821 | 0.857 | 0.456 | 0.816 | 0.809 ±0.000 |
| **Ensemble** | **0.838** | **0.865** | **0.476** | 0.828 | 0.791 |

## Ensemble — full test metrics (max-F1 threshold)
Acc 0.8653 · Precision 0.4257 · Recall/Sensitivity 0.5564 · F1 0.4763 · Specificity 0.8600 · AUC-ROC 0.8377 · AUC (5-dx) 0.8283.

## Cross-dataset CheXpert → NIH (7 common diseases)
EfficientNet 0.791 · DenseNet 0.799 · Rad-DINO 0.792 · **Ensemble 0.809**.

---

# PART 8 — ALL FILES (in `C:\AyShCXR` / `/workspace`)

**Training (CheXpert, the final pipeline):** `chexpert_efficientnet_train.py`, `chexpert_densenet_train.py`, `chexpert_raddino_train.py`
**NIH:** `nih_efficientnet_train.py`, `build_nih_csv.py`, `fix_nih_paths.py`, `resize_nih.py`, `make_nih_clean.py`
**Comparison harness (earlier):** `train_model.py`, `make_comparison_table.py`
**Evaluation:** `ensemble_chexpert.py`, `save_chexpert_preds.py`, `cv_chexpert.py`, `cross_dataset_eval.py`, `nih_test_eval.py`
**Table & report (laptop):** `build_chexpert_table.py`, `colab_chexpert_table.py`, `build_docx.js`
**Data prep (earlier):** `build_chexpert_csv.py`, `chexpert_train.py` (canonical Phase-2 file), `build_and_train_demo.py` (proven NIH template)
**Outputs:** `chexpert_preds.npz`, `cv_results.csv`, `crossdataset_results.csv`, `CheXpert_Model_Comparison.csv`, `roc_curves.png`, `auc_bar.png`
**Reports:** `CheXpert_Project_Report.md`, `CheXpert_Project_Report_FULL.docx` (the final Word report), `PROJECT_COMPLETE_SUMMARY.md` (this file)
**Checkpoints (server, never delete):** `chexpert_{efficientnet_b4,densenet121,rad_dino}_best_auc*.pth`
**Memory (persists across sessions):** `chexpert_plus_debugging.md`, `raddino_install_fix.md`, `project_ayshcxr.md`, `feedback_safety_rules.md`, `user_profile.md`

---

# PART 9 — KEY DECISIONS & FINDINGS (the "why")

1. **Impression labels, not full-report labels** — the single fix that made CheXpert Plus learnable (0.50 → 0.84).
2. **Performance is label-limited, not capacity-limited** — 3 architectures converge to ~0.82 because CheXBERT labels agree with radiologists at only F1≈0.44. This is the central scientific finding.
3. **Ensembling adds ~+1%** — uncorrelated errors cancel.
4. **GMP+GAP adds ~+0.5%** on the CNNs.
5. **Rad-DINO is the most data-efficient** (leads CV) — medical pretraining shines under limited training.
6. **Strong cross-dataset generalization** (~0.03 drop CheXpert→NIH) — the model learned real features.
7. **No overfitting** — Train ≈ Val ≈ Test throughout.
8. **High Acc/Spec, moderate Prec/F1** — the expected signature of imbalanced multi-label detection.

---

# PART 10 — HONEST CAVEATS (what NOT to claim)

- **NOT state-of-the-art.** CheXpert SOTA (~0.90–0.93) is on the 5-disease subset with **radiologist labels** and the **official test set**. Your 0.84 is all-14, CheXBERT labels, patient-split — a harder, non-identical setting. **Frame as a rigorous comparison study, not a "best model."**
- **Do not headline the micro-AUC 0.92** — it's inflated by easy negatives. The macro 0.838 is the honest number.
- **The FiLM 0.95–0.97 is label leakage** — never present it as a model AUC.
- **CheXpert Plus is research-only** — a deployed commercial product needs retraining on licensed/locally-collected data.
- **The 3-fold CV is reduced** (10 ep/90k) — report it transparently; it measures stability, not full-data performance.
- A real **clinical system** needs regulatory approval (CDSCO) + prospective validation on local PHC images — the model is a strong prototype foundation, not a certified product.

---

# PART 11 — STATUS: DONE vs LEFT

## ✅ DONE (every professor requirement)
- [x] Full metrics table — Acc/Prec/Recall/F1/Sensitivity/Specificity × Train/Val/Test
- [x] AUC-ROC (all-14 + 5-disease subset)
- [x] Cross-validation (true 3-fold, stable)
- [x] Rad-DINO transformer comparison (CNN vs transformer)
- [x] Cross-dataset CheXpert → NIH (excluding the 4 non-common diseases)
- [x] No MCC (excluded per professor)
- [x] Grad-CAM (done in an earlier phase, professor-approved)
- [x] Ensemble (headline 0.838 in-domain, 0.809 cross-dataset)
- [x] ROC curves + AUC bar chart
- [x] Word report (`CheXpert_Project_Report_FULL.docx`) with embedded graphs + landscape full table + cross-dataset section

## ⬜ OPTIONAL / FUTURE
- Report polish: Abstract, Table of Contents, References, embed the Grad-CAM figure.
- Legitimate FiLM (real symptoms/demographics, no leakage) — a future novel contribution.
- Cleaner radiologist labels or the official CheXpert test set → higher meaningful AUC.
- System/deployment: wire `app.py` to serve the new ensemble + Grad-CAM + MC-Dropout uncertainty; regulatory + local-data path for a real product.

---

# PART 12 — HOW TO REPRODUCE (quick reference)

1. Train 3 models: `python chexpert_{efficientnet,densenet,raddino}_train.py` (server, GPU) → `chexpert_<arch>_best_auc*.pth`.
2. Ensemble: `python ensemble_chexpert.py`.
3. Save predictions: `python save_chexpert_preds.py` → `chexpert_preds.npz`.
4. Cross-validation: `python cv_chexpert.py` → `cv_results.csv`.
5. Cross-dataset: `python cross_dataset_eval.py` → `crossdataset_results.csv`.
6. Build table + graphs (laptop): `python build_chexpert_table.py` (needs `chexpert_preds.npz` + `cv_results.csv`).
7. Build Word report (laptop): `node build_docx.js` → `CheXpert_Project_Report_FULL.docx`.

**Rad-DINO env (never touch torch):** `transformers==4.57.6` + `huggingface_hub==0.36.2`.

---

*Compiled 2026-06-22. This file (`PROJECT_COMPLETE_SUMMARY.md`) is the single source of truth for the AyShCXR project from inception through the completed CheXpert Plus comparative study.*
