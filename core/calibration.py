# calibration.py
# AyShCXR — probability calibration. 2026-08-10
# by Subhrakant Sethi & Ayush Singh
#
# THE PROBLEM
#   When the app says "70%", is it right 70 times out of 100? Neural networks
#   are systematically overconfident, so usually not. In a PHC that number
#   decides whether a patient is sent on a three-hour bus journey to a district
#   hospital. A miscalibrated 70% wastes a family's day or sends someone home
#   who should have gone.
#
#   Calibration does NOT change which finding ranks highest, so AUC is
#   unchanged by construction. It changes what the number MEANS.
#
# THE METHOD — per-disease temperature scaling
#   Fit one scalar T per finding on held-out validation data:
#       p_calibrated = sigmoid(logit(p_raw) / T)
#   T > 1 softens overconfident predictions; T < 1 sharpens underconfident ones.
#   Temperature scaling is monotonic, so ranking and AUC are provably preserved.
#   Fitted on VALIDATION only — fitting on test would be self-deception.
#
# MEASUREMENT — Expected Calibration Error (ECE)
#   Bin predictions by confidence, compare predicted vs observed frequency in
#   each bin, average the gaps weighted by bin size. Lower is better.
#
# Self-test / fit:  python core/calibration.py

import json
import os
import numpy as np

from disease_ontology import safe_to_canonical

CALIB_PATH = "results/calibration.json"
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE for one disease. 0 = perfectly calibrated."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        conf = probs[m].mean()          # what the model claimed
        acc = labels[m].mean()          # what actually happened
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def fit_temperature(probs, labels, lo=0.25, hi=8.0, steps=400):
    """Find T minimising negative log-likelihood.

    Grid search rather than gradient descent: one scalar, 400 points, runs in
    milliseconds, and cannot fail to converge.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if len(np.unique(labels)) < 2:
        return 1.0
    z = _logit(probs)
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(lo, hi, steps):
        p = np.clip(_sigmoid(z / t), EPS, 1 - EPS)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def fit_from_npz(npz_path="results/chexpert_preds.npz",
                 model_key="densenet121", out_path=CALIB_PATH):
    """Fit per-disease temperatures on the validation split and report the
    before/after effect on both ECE and AUC."""
    d = np.load(npz_path)
    labels = [str(x) for x in d["labels"]]
    pv, yv = d[f"p_{model_key}_val"], d["y_val"]
    pt, yt = d[f"p_{model_key}_test"], d["y_test"]

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    temps, rows = {}, []
    for i, name in enumerate(labels):
        # uncertain CheXpert labels were stored as 0.5; exclude them — a soft
        # label is not a ground-truth event and would corrupt the fit
        mv = np.isin(yv[:, i], (0.0, 1.0))
        mt = np.isin(yt[:, i], (0.0, 1.0))
        if mv.sum() < 50 or len(np.unique(yv[mv, i])) < 2:
            temps[name] = 1.0
            continue

        t = fit_temperature(pv[mv, i], yv[mv, i])
        temps[name] = t

        before = expected_calibration_error(pt[mt, i], yt[mt, i])
        after = expected_calibration_error(
            _sigmoid(_logit(pt[mt, i]) / t), yt[mt, i])
        auc_b = auc_a = None
        if roc_auc_score and len(np.unique(yt[mt, i])) > 1:
            auc_b = roc_auc_score(yt[mt, i], pt[mt, i])
            auc_a = roc_auc_score(yt[mt, i], _sigmoid(_logit(pt[mt, i]) / t))
        rows.append((name, t, before, after, auc_b, auc_a))

    payload = {"model": model_key, "temperatures": temps,
               "fitted_on": "validation split", "n_val": int(len(yv))}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return rows, out_path


# ── runtime use ─────────────────────────────────────────────────────────────
_TEMPS = None


def load(path=CALIB_PATH):
    global _TEMPS
    if _TEMPS is None:
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)["temperatures"]
            # the file is fitted under CheXpert spellings ("Pleural Effusion")
            # but the app looks findings up by canonical name ("Effusion");
            # without this, those findings were silently left uncalibrated
            _TEMPS = {safe_to_canonical(k) or k: t for k, t in raw.items()}
        except Exception:
            _TEMPS = {}
    return _TEMPS


def calibrate(disease, prob, path=CALIB_PATH):
    """Apply the fitted temperature. Unknown findings pass through unchanged —
    never guess a correction you have not measured."""
    t = load(path).get(safe_to_canonical(disease) or disease)
    if not t or abs(t - 1.0) < 1e-3:
        return float(prob)
    return float(_sigmoid(_logit(np.array([prob])) / t)[0])


def calibrate_all(probs, path=CALIB_PATH):
    return {d: calibrate(d, p, path) for d, p in probs.items()}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve()
    ROOT = next((p for p in HERE.parents if (p / ".ayshcxr_root").exists()), None)
    if ROOT:
        os.chdir(ROOT)

    print("=" * 78)
    print("  Probability calibration — per-disease temperature scaling")
    print("=" * 78)

    rows, path = fit_from_npz()
    print(f"\n{'finding':<26}{'T':>6}{'ECE before':>12}{'ECE after':>11}"
          f"{'AUC before':>12}{'AUC after':>11}")
    print("-" * 78)
    eb = ea = 0.0
    n = 0
    for name, t, b, a, ab, aa in rows:
        bs = f"{b:.4f}" if b is not None else "  -  "
        as_ = f"{a:.4f}" if a is not None else "  -  "
        abs_ = f"{ab:.4f}" if ab is not None else "  -  "
        aas = f"{aa:.4f}" if aa is not None else "  -  "
        mark = "  <-- improved" if (b and a and a < b) else ""
        print(f"{name:<26}{t:>6.2f}{bs:>12}{as_:>11}{abs_:>12}{aas:>11}{mark}")
        if b is not None and a is not None:
            eb += b; ea += a; n += 1

    if n:
        print("-" * 78)
        print(f"{'MEAN':<26}{'':>6}{eb/n:>12.4f}{ea/n:>11.4f}")
        print(f"\ncalibration error reduced by "
              f"{(1 - ea/eb) * 100:.1f}% on the held-out test split")

    aucs_b = [r[4] for r in rows if r[4] is not None]
    aucs_a = [r[5] for r in rows if r[5] is not None]
    if aucs_b:
        print(f"macro AUC before {np.mean(aucs_b):.4f}  "
              f"after {np.mean(aucs_a):.4f}  "
              f"(delta {np.mean(aucs_a) - np.mean(aucs_b):+.6f})")
        print("AUC is unchanged by design — temperature scaling is monotonic, "
              "so it cannot reorder predictions.")

    print(f"\nwrote {path}")
    demo = {"Pneumonia": 0.90, "Cardiomegaly": 0.75, "Hernia": 0.30}
    print("\nruntime example:")
    for d, p in demo.items():
        print(f"   {d:16s} raw {p:.2f} -> calibrated {calibrate(d, p):.3f}")
    print("\n" + "=" * 78)
