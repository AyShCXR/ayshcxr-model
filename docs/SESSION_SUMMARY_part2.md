# AyShCXR — Session Summary, Part 2 (Reference-Baseline + Debugging Saga)

*Date: 2026-06-22. Continues from the first full session summary (CheXpert comparison study + report).*

---

## 0. Where this session started
The previous summary ended with: CheXpert comparison study finished, `CheXpert_Project_Report_FULL.docx` written, GPU expiring. This session began with a **pivot in goals** and then got hijacked (productively) by a **professor requirement** and a **long GPU debugging saga**. End state: the professor's "ready for the paper" gate is **cleared**, and all trained models are safely on the laptop.

---

## 1. The pivot (start of session)
Subhrakant wanted to stop "satisfying the professor" and start building something genuinely better: **pick the best model as a baseline, then improve day-by-day to raise AUC and address drawbacks raised in research papers.**

I entered **plan mode** and asked 3 decision questions. Answers:
- **Compute going forward:** *"Tonight only, then laptop"* — one last GPU night, then laptop-only.
- **North star:** *"all 4"* (beat the label ceiling, robustness/trust, both sequenced, single headline AUC).
- **Drawbacks to cover:** *Label noise / uncertainty*, *Shortcut learning*, *Calibration & confidence* (NOT cross-dataset generalization).

Key honest framing I gave: the ~0.82 AUC is **label-limited, not model-limited** (CheXBERT labels agree with radiologists at only F1≈0.44), so raw "improve AUC" has a ceiling; the real gains live in attacking *why* the ceiling exists.

---

## 2. The professor (Sir) meeting — the real driver of this session
Subhrakant visited Sir, who (decoded from the Hindi) demanded:
> **Prove the model is genuinely better by comparing it against a reference/baseline model on the SAME train/validation/test data, the SAME sample size, via 3-fold CV (test set set aside).** Reproduce the *reference paper's* model at the same sample size; show 3-fold average performance. If it holds → ready for the paper.

Subhrakant was overwhelmed: **he never formally targeted a reference paper** — he just built a deployable system and read papers for ideas.

### What I clarified
- Sir's ask is **standard, normal science** — a fair baseline comparison, not "beat the world record."
- You don't need to have pre-chosen a reference; you pick a recognized baseline now and run it under identical conditions.
- **Stay on CheXpert, not NIH** (everything's already built for CheXpert; NIH switch = weeks of redo; CheXpert also has the radiologist gold set, NIH doesn't).

### The leaderboard myth (user sent the Stanford CheXpert leaderboard, 0.930 entries)
I killed the panic: the **0.930 is NOT comparable** to their 0.84 because it's:
- only **5 diseases** (vs their 14),
- graded on **radiologist gold labels** (vs their auto-labels),
- an **ensemble of ~30 checkpoints** by pro teams (vs a single model).
Putting 0.84 next to 0.930 is apples-to-oranges — exactly the "manipulation" Sir was warning against.

### The reference paper resolution
The user shared `1901.07031v1 (1).pdf`. **That IS the CheXpert paper (Irvin & Rajpurkar et al., 2019)** — the one linked on the leaderboard ("READ THE PAPER"). Its model is **DenseNet-121**. So:
- **Reference paper** = CheXpert paper (the paper behind the dataset they use → unbeatable choice).
- **Reference model** = DenseNet-121 = the same standard baseline I'd already recommended.
"Reference paper" and "baseline model" collapse into one thing — no scary external paper needed.

Clarified the conceptual confusion: **"their model" = the architecture (DenseNet-121, built into PyTorch), NOT their trained weights.** We rebuild it and train on our data — that's the fair comparison. Their 320px/3-epoch/0.90 numbers are NOT used.

### What I read from the CheXpert paper (via pypdf → `_chexpert_paper_text.txt`)
- Model: **DenseNet-121** (chosen over ResNet-152, Inception-v4, SE-ResNeXt101).
- Input **320×320**, **Adam** lr 1e-4 (β1=0.9, β2=0.999), batch **16**, **3 epochs**.
- Eval on **5 diseases** (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion).
- **200-study** validation (3-radiologist consensus), **500-study** test (5-radiologist consensus).
- **Uncertainty approaches:** U-Ignore, U-Zeros, U-Ones, U-SelfTrained, U-MultiClass (different best per disease).
- Validation AUCs (Table 3) best-per-disease ≈ 0.858 / 0.854 / 0.939 / 0.941 / 0.936 → **~0.90 mean**, from a **30-checkpoint ensemble**.

---

## 3. The plan (approved)
Plan file: `C:\Users\SUBHRAKANT SETHI\.claude\plans\quiet-sleeping-meerkat.md`.
- Reproduce **vanilla DenseNet-121** (GAP-only + single Linear head — i.e., the CheXpert-paper model, WITHOUT our GMP+GAP/complex-head improvements) through the **existing 3-fold CV harness**, on the **identical** 90k sample / seed-42 / `GroupKFold(3)` folds.
- Because the harness is **deterministic**, the 3 existing model results in `cv_results.csv` are already on the same folds → only **one new run** (the vanilla baseline) needed.
- Output one head-to-head table; test set untouched.

---

## 4. Scripts created
- **`cv_reference_baseline.py`** (server/GPU) — vanilla DenseNet-121 through 3-fold CV. Mirrors `cv_chexpert.py` byte-for-byte (CLEAN_CSV `/workspace/chexpert_clean.csv`, SEED 42, SUBSET 90000, GroupKFold(3), EPOCHS 10, IMG_SIZE 380, BATCH 64, same FocalLoss / transforms / AdamW backbone-6e-5 + head-2e-4 / cosine). Only the architecture differs. Appends `densenet121_vanilla` row to `cv_results.csv`. (New file — did NOT touch the working `cv_chexpert.py`; never touches any `.pth`.)
- **`build_baseline_table.py`** (laptop) — reads `cv_results.csv`, maps arch keys → human labels, orders "ours" above the reference, writes `Baseline_vs_Ours.csv`. Smoke-tested; fixed a Windows **cp1252 console crash** (replaced `Δ`/`±`/`—` with ASCII, added `sys.stdout.reconfigure(utf-8)`).
- Backed up `cv_results.csv` → `cv_results_backup.csv` (per the always-backup safety rule).

---

## 5. THE DEBUGGING SAGA (the bulk of the session)
Running `cv_reference_baseline.py` on the server hit issue after issue. Full sequence:

1. **Run 1 (foreground):** trained fine to **79% of epoch 1 at 6.70 it/s** with `torch.compile` + `num_workers=4`. User pressed **Ctrl+C** to relaunch under `nohup`.
2. **Run 2 (nohup):** crashed instantly — **`/dev/shm` Bus error** ("DataLoader worker killed... insufficient shared memory"). Cause: the Ctrl+C of Run 1 **leaked shared memory**; the new run couldn't allocate.
3. **My first (over)correction:** removed `torch.compile` + set `num_workers=0`. User pushed back hard: *"don't remove torch.compile, find the actual error."* (Correct instinct.)
4. Restored compile; set **train workers=4, val workers=0** (matching the proven full-training recipe); had user clean shm (`pkill -9 -f ...` + `rm -rf /dev/shm/*`).
5. **Run 3 (nohup):** appeared "stuck" at `patient overlap 0`. `nvidia-smi`: **GPU idle (115–136W, 31°C)**, never warming.
6. **Foreground diagnostic run:** **WORKED** (reached 21%, 6.66 it/s) → proved the *code* was fine and `nohup` was the variable.
7. **nohup `-u` relaunch:** still hung at `patient overlap 0`; GPU still idle.
8. **The smoking gun — `ps aux`:** main PID **476% CPU for 33 minutes**, GPU idle, **~32 sleeping subprocesses**. Those are **TorchInductor's parallel compile-worker pool, deadlocked**. (`py-spy dump` failed — container blocks ptrace — but `ps` was conclusive.)

### ROOT CAUSE (the actual error)
**`torch.compile` (TorchInductor) deadlocks when the script is launched in the background via `nohup` — it spawns ~32 parallel compile-worker subprocesses that hang with no controlling TTY, so compilation never finishes and training never starts.** It worked yesterday because yesterday's training was **not** run under `nohup`. That's why "yesterday it wasn't a problem."

### THE FIX (keeps compile + full speed)
Set **`TORCHINDUCTOR_COMPILE_THREADS=1`** → serial compilation, no worker pool to deadlock.
- Added `os.environ["TORCHINDUCTOR_COMPILE_THREADS"]="1"` at the top of the script (before `import torch`), AND prefixed the launch command with it.
- Kept `torch.compile` + `num_workers=4` (train) / `0` (val).
- Added `python -u` (so `nohup` logs aren't buffered → no more "looks frozen"), a **per-epoch loss print**, and a **"first batch JIT-compiles ~2-3 min, GPU idle is normal"** message.

### Other server gotchas confirmed
- Always `pkill -9 -f <script>` + `rm -rf /dev/shm/*` before relaunch (killed DataLoader workers leak shm → Bus error).
- `python -u` is mandatory under `nohup` or output looks stuck.
- First-batch JIT compile = ~2–3 min of GPU-idle silence = normal.

**Final run: all 3 folds completed at full speed.** Saved to memory: `torch_compile_nohup_fix.md` (+ MEMORY.md index updated).

---

## 6. RESULTS — the reference-baseline comparison
Reference (vanilla DenseNet-121): **3-fold CV AUC = 0.7924 ± 0.0022** (folds 0.7950 / 0.7927 / 0.7896).

Final head-to-head table (`Baseline_vs_Ours.csv`, identical 90k folds, test set untouched):

| Model | 3-fold AUC | vs reference |
|---|---|---|
| **Rad-DINO (ours)** | **0.8089** ± 0.0003 | **+0.0165** ✅ |
| **DenseNet-121 + GMP+GAP (ours)** | **0.7983** ± 0.0010 | **+0.0059** ✅ |
| EfficientNet-B4 + GMP+GAP (ours) | 0.7644 ± 0.0013 | −0.0280 ✗ |
| DenseNet-121 *(CheXpert paper baseline)* | 0.7924 ± 0.0022 | — |

**2 of 3 models beat the reference.** Best (Rad-DINO, medical pretraining) **+1.65%**; the GMP+GAP enhancement beats the *same-backbone* vanilla DenseNet by **+0.59%** (a clean controlled win, larger than the ±0.001 noise). EfficientNet underperforms the baseline — reported honestly for transparency (not the proposed model).

### Honest framing given to Sir
> "I reproduced the CheXpert paper's DenseNet-121 under my identical 3-fold protocol. My proposed model (Rad-DINO) beats it by 1.65%, and my DenseNet+GMP+GAP enhancement beats the same vanilla DenseNet by 0.59% on identical folds. EfficientNet underperformed and is reported for completeness."

---

## 7. GPU banking before expiry (everything irreversible)
- Confirmed the new best checkpoints lived **only on the server** (not the laptop).
- Inventory found the **gold validation IMAGES exist** (`/chexpert_plus_png_412/PNG/valid`) but **NO radiologist `valid.csv`** anywhere → **gold-label eval not feasible**, skipped (it was optional bonus; gate already cleared).
- Downloaded + moved into `C:\AyShCXR`:
  - `chexpert_densenet121_best_auc0.8281_ep12.pth` (29 MB) — best / deployable model
  - `chexpert_efficientnet_b4_best_auc0.8226_ep16.pth` (71 MB)
  - `chexpert_rad_dino_best_auc0.8238_ep4.pth` (332 MB)
  - `chexpert_clean.csv` (28 MB), `cv_results.csv`, and `bank_tonight.tar` (461 MB bundle backup)
- **GPU can now expire safely — nothing left to run.**

---

## 8. Deliverables produced this session (in `C:\AyShCXR`)
- `cv_reference_baseline.py` — the reference-baseline trainer (with the COMPILE_THREADS fix)
- `build_baseline_table.py` — the head-to-head table builder
- `Baseline_vs_Ours.csv` — **the table to show Sir**
- `cv_results.csv` (now 4 rows incl. `densenet121_vanilla`) + `cv_results_backup.csv`
- 3 model checkpoints + `chexpert_clean.csv` + `bank_tonight.tar`
- `_chexpert_paper_text.txt` — extracted CheXpert paper text
- Plan: `…/.claude/plans/quiet-sleeping-meerkat.md`
- Memory: `torch_compile_nohup_fix.md`

---

## 9. Still open / next steps (all laptop-only now)
1. **Show Sir the table** + confirm one line: *"baseline = standard DenseNet-121, thik hai?"* (swap if he names another paper).
2. **Fold the baseline comparison into the report** (`.docx`).
3. **The improvement phase** Subhrakant wanted — calibration (temperature scaling), Grad-CAM / shortcut-learning analysis, label-noise handling — all runnable on the laptop with the banked checkpoints + `chexpert_clean.csv`.
4. **Gold-label AUC** (comparable to the paper's 0.90) is **blocked** — no radiologist `valid.csv` in the CheXpert-Plus download; would need the original CheXpert v1.0 valid labels.

---

*Bottom line: the professor's gate is cleared with a rigorous, honest, identical-conditions 3-fold comparison; the real bug (torch.compile deadlocking under nohup) was found and permanently fixed; and every trained model is safe on the laptop.*
