# Morning Plan — AyShCXR CheXpert Training

## Where we are
- Data is CONFIRMED PERFECT: images valid (saw montage), labels match reports, resize faithful.
- Problem: model fits TRAIN (~0.67 AUC) but NEVER generalizes (held-out ~0.52), across ~10 configs.
- Deepest insight: model learns each disease's BASE RATE + memorizes train, but learns NO
  generalizable features. Train loss == val loss == ~0.36 (below baseline = base-rate fit).

## STEP 1 — Run the diagnostic FIRST (5 min). This is the gate.
Upload overfit_test.py, then:
    python /workspace/overfit_test.py

It trains on 200 images and checks if the model can MEMORIZE them (train AUC).
- TRAIN AUC > 0.90  -> model + pipeline + optimization are FINE.
                       The issue is purely GENERALIZATION -> go to STEP 2 (full run justified).
- TRAIN AUC < 0.70  -> FUNDAMENTAL pipeline/optimization bug. Do NOT train.
                       Tell Claude the number; we debug the pipeline instead.

## STEP 2 — Only if overfit test passes (>0.90): launch full training
Main is already converted to the best config:
  - 3-channel input (standard CheXpert, full pretrained conv)
  - BCE loss (Claude will add pos_weight back before launch — ask)
  - Adam 1e-4, light augmentation, loose clip 5.0, full 175k images

    pkill -f train_rad_dino_chexpert
    rm -f /workspace/efficientnet_chexpert_checkpoint_epoch*.pth /workspace/efficientnet_chexpert_best_*.pth
    nohup python /workspace/train_rad_dino_chexpert.py 2>&1 | tee /workspace/raddino_run.log &
    tail -f /workspace/raddino_run.log

## STEP 3 — Go/No-Go at EPOCH 9 (~1.5 hours)
    tail -40 /workspace/raddino_run.log
- Val AUC climbing (0.55 -> 0.62 -> ...) -> WORKING. Let it finish (~7 hrs). Best model auto-saves.
- Still ~0.50 at epoch 9 -> STOP. We regroup (reference implementation / different approach).
  Do NOT waste remaining GPU days.

## After a successful run
- python /workspace/eval_full_metrics.py  -> professor's table (Acc/Prec/Recall/F1/Sens/Spec
  for train/val/test + AUC). Already updated to 3-channel.
- Then Rad-DINO run (transformer the professor wants).

## Files ready on server / to upload
- overfit_test.py         (diagnostic — run FIRST)
- train_rad_dino_chexpert.py (main, 3-channel)
- eval_full_metrics.py    (professor's metrics table)
- train_sanity.py         (quick experiments)
