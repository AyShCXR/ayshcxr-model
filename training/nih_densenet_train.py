# nih_densenet_train.py
# AyShCXR — NIH ChestX-ray14, DenseNet-121, proven GMP+GAP recipe.
# by Subhrakant Sethi & Ayush Singh
#
# SELF-CONTAINED SERVER SCRIPT. Does prep -> resize -> train in one pass and is
# resumable at every stage, because GPU sessions drop and access is short.
#
#   pip install -q kagglehub iterative-stratification
#   nohup python nih_densenet_train.py > nih_train.log 2>&1 &
#   tail -f nih_train.log
#
# Recipe is deliberately IDENTICAL to chexpert_densenet_train.py (which produced
# the best single model, 0.8281) except for the dataset and label set:
#   3-channel RGB, GMP+GAP dual pooling, Focal loss + per-disease weights,
#   2-group LLRD AdamW, cosine annealing, 18 epochs, AMP, no horizontal flip.
#
# EXPECTED RESULT: 0.79-0.82. NOT higher. Three architectures already converged
# to ~0.82 on an H100 -- the ceiling is label noise, not compute. The radiologist
# benchmark on this dataset is 0.778. Anything above ~0.83 means a bug (most
# likely a train/test patient leak), not a breakthrough.

import os, sys, glob, time, argparse
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")   # torch.compile deadlocks under nohup otherwise

import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy("file_system")     # small /dev/shm
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
WORK        = "/workspace"
CSV_PATH    = f"{WORK}/nih_clean.csv"
RESIZE_DIR  = f"{WORK}/nih_resized"
RESIZE_PX   = 412                    # = IMG_SIZE + 32, the first transform step
IMG_SIZE    = 380
BATCH       = 64
NUM_EPOCHS  = 18
SEED        = 42
VAL_FRAC    = 0.08

# NIH's own spelling uses underscores in some columns; canonical form has spaces.
NIH_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
              "Mass", "Nodule", "Pneumonia", "Pneumothorax",
              "Consolidation", "Edema", "Emphysema", "Fibrosis",
              "Pleural Thickening", "Hernia"]

# sqrt-normalised inverse frequency; Pneumonia weighted up (rural mortality),
# Hernia highest (rarest). Carried over from build_and_train_demo.py.
DISEASE_WEIGHTS = {
    "Atelectasis": 0.57, "Cardiomegaly": 1.21, "Effusion": 0.53,
    "Infiltration": 0.50, "Mass": 0.83, "Nodule": 0.79, "Pneumonia": 2.00,
    "Pneumothorax": 0.87, "Consolidation": 0.93, "Edema": 1.33,
    "Emphysema": 1.27, "Fibrosis": 1.56, "Pleural Thickening": 1.10,
    "Hernia": 3.00,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── stage 1: build the label CSV ────────────────────────────────────────────
def build_csv(search_root: str) -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"[prep] reusing {CSV_PATH} ({len(df):,} rows)")
        return df

    entry = next(iter(glob.glob(os.path.join(search_root, "**", "Data_Entry_2017*.csv"),
                                recursive=True)), None)
    if entry is None:
        sys.exit(f"[prep] FATAL: Data_Entry_2017.csv not found under {search_root}\n"
                 f"       Has the kagglehub download finished? Check nih_download.log")
    print(f"[prep] labels: {entry}")

    print("[prep] indexing png files (this takes a minute on NFS) ...")
    paths = {os.path.basename(p): p
             for p in glob.glob(os.path.join(search_root, "**", "*.png"), recursive=True)}
    print(f"[prep] indexed {len(paths):,} images")
    if len(paths) < 100_000:
        print(f"[prep] WARNING: expected ~112,120 images, found {len(paths):,}. "
              f"Download may still be running.")

    df = pd.read_csv(entry)
    df["full_path"] = df["Image Index"].map(paths)
    miss = int(df["full_path"].isna().sum())
    df = df.dropna(subset=["full_path"]).reset_index(drop=True)
    print(f"[prep] matched {len(df):,} rows ({miss:,} unmatched)")

    # "Finding Labels" is a |-separated string, e.g. "Effusion|Infiltration"
    findings = df["Finding Labels"].fillna("")
    for lab in NIH_LABELS:
        nih_spelling = lab.replace(" ", "_")          # Pleural Thickening -> Pleural_Thickening
        df[lab] = findings.str.contains(nih_spelling, regex=False).astype("float32")

    df["patient_id"] = df["Patient ID"].astype(str)
    keep = ["Image Index", "full_path", "patient_id"] + NIH_LABELS
    df = df[keep]
    df.to_csv(CSV_PATH, index=False)
    print(f"[prep] wrote {CSV_PATH}")
    print("[prep] positives per label:\n" +
          df[NIH_LABELS].sum().astype(int).to_string())
    return df


# ── stage 2: pre-resize 1024px -> 412px (~3x faster epochs) ─────────────────
try:    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError: _RESAMPLE = Image.LANCZOS


def _resize_one(args):
    """Module-level so multiprocessing can pickle it. Idempotent: an existing
    destination is left alone, which is what makes the stage resumable."""
    name, src = args
    dst = os.path.join(RESIZE_DIR, name)
    if os.path.exists(dst):
        return dst
    try:
        Image.open(src).convert("L").resize((RESIZE_PX, RESIZE_PX),
                                            _RESAMPLE).save(dst, "PNG")
        return dst
    except Exception as e:
        print("[resize] FAIL", src, e)
        return None


def resize_all(df: pd.DataFrame) -> pd.DataFrame:
    from multiprocessing import Pool
    os.makedirs(RESIZE_DIR, exist_ok=True)

    todo = [(n, p) for n, p in zip(df["Image Index"], df["full_path"])
            if not os.path.exists(os.path.join(RESIZE_DIR, n))]
    print(f"[resize] {len(df) - len(todo):,} already done | {len(todo):,} to do")

    if todo:
        t0 = time.time()
        with Pool(8) as pool:
            list(tqdm(pool.imap_unordered(_resize_one, todo, chunksize=64),
                      total=len(todo), desc="resize"))
        print(f"[resize] took {(time.time() - t0)/60:.1f} min")

    df = df.copy()
    df["full_path"] = [os.path.join(RESIZE_DIR, n) for n in df["Image Index"]]
    ok = df["full_path"].map(os.path.exists)
    print(f"[resize] usable images: {int(ok.sum()):,} / {len(df):,}")
    return df[ok].reset_index(drop=True)


# ── model: identical to chexpert_densenet_train.py ──────────────────────────
def complex_head(in_f, n=14):
    return nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4),
                         nn.Linear(in_f, 512), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(512, n))


class CNNDualPool(nn.Module):
    """DenseNet-121 with GMP+GAP dual pooling (worth ~+0.5% AUC on CNNs)."""
    def __init__(self, n=14):
        super().__init__()
        bb = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features   = bb.features
        self.post       = nn.ReLU(inplace=True)
        in_f            = bb.classifier.in_features
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.gmp        = nn.AdaptiveMaxPool2d(1)
        self.classifier = complex_head(in_f, n)

    def forward(self, x):
        f = self.post(self.features(x))
        return self.classifier(self.gap(f).flatten(1) + self.gmp(f).flatten(1))


class FocalLoss(nn.Module):
    def __init__(self, g=2.0, a=0.75, s=0.1, w=None):
        super().__init__(); self.g, self.a, self.s, self.w = g, a, s, w
    def forward(self, x, t):
        x = x.float(); t = t.float().clamp(0, 1)
        ts = t * (1 - self.s) + 0.5 * self.s
        bce = F.binary_cross_entropy_with_logits(x, ts, reduction="none").clamp(max=50)
        pt  = torch.exp(-bce)
        fl  = self.a * (1 - pt) ** self.g * bce
        if self.w is not None:
            fl = fl * self.w.to(x.device)
        return fl.mean()


class DS(Dataset):
    def __init__(self, d, tf): self.d = d.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        try:    img = Image.open(r["full_path"]).convert("RGB")
        except Exception: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(img), torch.tensor(r[NIH_LABELS].to_numpy(dtype="float32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=WORK,
                    help="folder containing the kagglehub NIH download")
    ap.add_argument("--skip-resize", action="store_true")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = ap.parse_args()

    print("=" * 62)
    print("  AyShCXR — NIH ChestX-ray14 | DenseNet-121 | GMP+GAP")
    print(f"  device {device} | img {IMG_SIZE} | batch {BATCH} | epochs {args.epochs}")
    print("=" * 62)

    df = build_csv(args.data_root)
    if not args.skip_resize:
        df = resize_all(df)

    # patient-level split — NIH has multiple images per patient; an image-level
    # split leaks the same patient into train and val and inflates AUC.
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRAC, random_state=SEED)
                  .split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    overlap = len(set(train_df["patient_id"]) & set(val_df["patient_id"]))
    print(f"\ntrain {len(train_df):,} | val {len(val_df):,} | patient overlap {overlap}")
    assert overlap == 0, "patient leak between train and val"

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)), T.RandomCrop(IMG_SIZE),
        T.RandomRotation(15), T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomPerspective(0.1, p=0.3), T.ColorJitter(0.2, 0.2),
        T.GaussianBlur(3, sigma=(0.1, 1.0)), T.ToTensor(), norm])
        # NO horizontal flip — the heart is left-sided; flipping fabricates dextrocardia
    val_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])

    train_loader = DataLoader(DS(train_df, train_tf), batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(DS(val_df, val_tf), batch_size=BATCH * 2, shuffle=False,
                              num_workers=0, pin_memory=True)   # shm-safe

    model = CNNDualPool(len(NIH_LABELS)).to(device)
    try:
        model = torch.compile(model); print("torch.compile applied")
    except Exception as e:
        print("compile skipped:", e)

    bb_p = [p for n, p in model.named_parameters() if "classifier" not in n]
    hd_p = [p for n, p in model.named_parameters() if "classifier" in n]
    opt = torch.optim.AdamW([{"params": bb_p, "lr": 6e-5, "weight_decay": 1e-5},
                             {"params": hd_p, "lr": 2e-4, "weight_decay": 1e-4}])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    crit = FocalLoss(w=torch.tensor([DISEASE_WEIGHTS[d] for d in NIH_LABELS],
                                    dtype=torch.float32))

    def run_val():
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                with torch.amp.autocast("cuda"):
                    p = torch.sigmoid(model(imgs.to(device, non_blocking=True)))
                P.append(p.float().cpu().numpy()); Y.append(labs.numpy())
        P, Y = np.vstack(P), np.vstack(Y)
        per = {d: roc_auc_score(Y[:, i], P[:, i])
               for i, d in enumerate(NIH_LABELS) if len(np.unique(Y[:, i])) > 1}
        return float(np.mean(list(per.values()))), per

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = nb = 0
        for imgs, labs in tqdm(train_loader, desc=f"ep{ep:02d}"):
            imgs = imgs.to(device, non_blocking=True); labs = labs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
            loss = crit(logits.float(), labs)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()

        auc, per = run_val()
        flag = "  BEST" if auc > best else ""
        print(f"  ep{ep:02d} | loss {tot/nb:.4f} | val-AUC {auc:.4f}{flag}", flush=True)
        if auc > best:
            best = auc
            out = f"{WORK}/nih_densenet121_best_auc{auc:.4f}_ep{ep}.pth"
            torch.save(model.state_dict(), out)
            pd.DataFrame({"disease": list(per), "auc": list(per.values())}) \
              .to_csv(f"{WORK}/nih_densenet121_per_disease.csv", index=False)
            print(f"       saved {out}", flush=True)

    print(f"\nDONE | best val-AUC {best:.4f}")
    print("Checkpoint keys carry a '_orig_mod.' prefix from torch.compile — "
          "the model registry already handles this via strip_prefix.")
    if best > 0.84:
        print("\n!! {:.4f} is above the plausible ceiling for NIH all-14. "
              "Check for a patient leak before believing it.".format(best))


if __name__ == "__main__":
    main()
