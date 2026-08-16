# image_quality.py
# AyShCXR — radiographic technique assessment. 2026-08-10
# by Subhrakant Sethi & Ayush Singh
#
# WHY THIS EXISTS
#   In a rural PHC the machine is old and the operator may have had two days of
#   training. Bad films are common, and a bad film manufactures disease:
#     rotation          -> heart projects wider  -> FALSE cardiomegaly
#     poor inspiration  -> bases look crowded    -> FALSE basal opacity
#     under-penetration -> everything looks white-> FALSE infiltration
#     over-penetration  -> small nodules vanish  -> MISSED lesion
#   The model has no idea any of this happened. It reads a rotated film and
#   confidently reports cardiomegaly. Warning the operator BEFORE they act on
#   that is worth more than a percentage point of AUC.
#
# ── HONEST LIMITATIONS — read before trusting any output ────────────────────
#   These are IMAGE HEURISTICS, not anatomical landmark detection. A proper
#   implementation locates the clavicles, spinous processes, diaphragm and ribs
#   with a trained segmentation model. This module infers from intensity and
#   symmetry instead, because that needs no extra model and no GPU.
#
#   Consequences you must accept:
#     * Rotation detection is approximate — it measures lung-field symmetry,
#       which rotation affects but so do large effusions and pneumonectomy.
#     * Inspiration is inferred from diaphragm height, not by counting ribs.
#     * A "PASS" here does NOT certify a technically perfect film.
#
#   Every check therefore returns advisory severity, never a hard verdict, and
#   the app must present these as "possible technical issue", not as fact.
#   They are designed to be over-cautious: a false warning costs a second look,
#   a missed warning costs a wrong diagnosis.
#
# Self-test:  python core/image_quality.py [image.png]

import numpy as np
from PIL import Image

# ── thresholds — MEASURED, not guessed ─────────────────────────────────────
# Calibrated against 400 real VinDr-CXR radiographs (2026-08-10). The first
# draft used invented thresholds and flagged 98% of normal films as blurred and
# 43% as clipped, which is worse than useless — a warning that fires on
# everything trains the operator to ignore warnings.
#
# These sit at population percentiles so only genuine outliers trigger:
#     warn   ~= p5 / p95   -> roughly 5% of normal films
#     severe ~= p1 / p99   -> roughly 1% of normal films
#
# Measured distribution (n=400):
#   rotation     p50 0.072   p95 0.243   p99 0.335
#   mean         p1  0.331   p50 0.570   p99 0.735
#   std          p5  0.182   p50 0.253
#   sharpness    p1  6.77    p50 18.32   p95 37.53
#   dark_clip    p50 0.040   p95 0.184   p99 0.218   <- black borders are NORMAL
ROTATION_WARN      = 0.24
ROTATION_SEVERE    = 0.34
DARK_MEAN          = 0.39    # below this = over-penetrated
DARK_MEAN_SEVERE   = 0.33
BRIGHT_MEAN        = 0.69    # above this = under-penetrated
BRIGHT_MEAN_SEVERE = 0.74
LOW_CONTRAST_STD   = 0.18
BLUR_LAPLACIAN     = 9.1     # p5 of real films
BLUR_SEVERE        = 6.8     # p1
DARK_CLIP_WARN     = 0.19    # p95 — VinDr images carry black borders
BRIGHT_CLIP_WARN   = 0.03

# The inspiration check is DISABLED. Measurement showed the "largest downward
# intensity gradient" lands on the image border rather than the diaphragm in a
# large fraction of films (p75 = 0.988 of image height, i.e. the bottom edge),
# so it flagged 33% of normal radiographs. Assessing inspiration properly means
# counting posterior ribs above the diaphragm, which needs anatomical landmark
# detection this module deliberately avoids. Shipping a broken check is worse
# than shipping none: it teaches the operator to dismiss warnings.
ENABLE_INSPIRATION = False


def _to_gray_array(img):
    """PIL image -> float32 array in [0,1]."""
    if img.mode != "L":
        img = img.convert("L")
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a


def _laplacian_variance(a):
    """Blur proxy. Sharp images have high edge energy."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = a.shape
    if h < 5 or w < 5:
        return 0.0
    # valid-region convolution without scipy
    out = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4.0 * a[1:-1, 1:-1]) if False else None
    out = (k[0, 1] * a[:-2, 1:-1] + k[2, 1] * a[2:, 1:-1]
           + k[1, 0] * a[1:-1, :-2] + k[1, 2] * a[1:-1, 2:]
           + k[1, 1] * a[1:-1, 1:-1])
    return float(np.var(out) * 1e4)


def _spine_column(a):
    """Approximate the midline.

    On a frontal chest film the mediastinum and spine are the brightest
    vertical band through the middle third. Column-summing the central band and
    taking the brightest column is a crude but stable midline estimate.
    """
    h, w = a.shape
    band = a[int(h * 0.15):int(h * 0.65), :]
    col = band.mean(axis=0)
    lo, hi = int(w * 0.30), int(w * 0.70)      # search only the central 40%
    return int(lo + np.argmax(col[lo:hi]))


def check_rotation(a):
    """Compare lung-field 'darkness mass' either side of the midline.

    Air-filled lung is dark. A rotated patient presents one hemithorax more
    obliquely, changing its projected area. Genuine unilateral pathology (large
    effusion, collapse) produces the same signature, so this is reported as
    'asymmetry', not definitively as rotation.
    """
    h, w = a.shape
    mid = _spine_column(a)
    roi = a[int(h * 0.20):int(h * 0.75), :]
    darkness = 1.0 - roi                       # lung = high value
    left = float(darkness[:, :mid].sum())
    right = float(darkness[:, mid:].sum())
    total = left + right
    asym = abs(left - right) / total if total > 0 else 0.0

    if asym >= ROTATION_SEVERE:
        sev, msg = "severe", ("Marked left-right asymmetry. The patient may be "
                              "rotated, or there may be significant unilateral "
                              "pathology. Cardiomegaly and unilateral findings "
                              "are unreliable on this film.")
    elif asym >= ROTATION_WARN:
        sev, msg = "warn", ("Some left-right asymmetry. If the patient was "
                            "rotated, heart width may be exaggerated.")
    else:
        sev, msg = "pass", "Hemithorax symmetry within normal range."
    return {"check": "rotation", "severity": sev, "value": round(asym, 4),
            "message": msg, "midline_col": mid}


def check_inspiration(a):
    """Estimate diaphragm height from the strongest lower-field horizontal edge.

    Shallow inspiration raises the diaphragm, crowds the lung bases and
    manufactures basal opacity. Radiologists count posterior ribs; this uses
    diaphragm position, which is cruder but requires no rib detection.
    """
    h, w = a.shape
    core = a[:, int(w * 0.20):int(w * 0.80)]
    rows = core.mean(axis=1)
    grad = np.diff(rows)
    lo = int(h * 0.35)                          # diaphragm never above this
    if lo >= len(grad):
        return {"check": "inspiration", "severity": "unknown", "value": None,
                "message": "Image too small to assess."}
    idx = lo + int(np.argmax(grad[lo:]))        # brightest downward transition
    frac = idx / h

    if frac < 0.50:
        sev, msg = "severe", ("Diaphragm appears high — inspiration may be "
                              "shallow. Lung bases will look crowded and can "
                              "mimic basal opacity or cardiomegaly.")
    elif frac < DIAPHRAGM_LOW:
        sev, msg = "warn", ("Inspiration may be sub-optimal. Consider repeating "
                            "with a full breath held.")
    else:
        sev, msg = "pass", "Inspiration appears adequate."
    return {"check": "inspiration", "severity": sev, "value": round(frac, 4),
            "message": msg}


def check_exposure(a):
    """Penetration and contrast from the intensity histogram."""
    mean = float(a.mean())
    std = float(a.std())
    dark_clip = float((a < 0.02).mean())
    bright_clip = float((a > 0.98).mean())

    issues = []
    sev = "pass"
    if mean < DARK_MEAN_SEVERE:
        issues.append("Film is markedly over-penetrated (too dark). Small "
                      "nodules and subtle lesions are likely invisible.")
        sev = "severe"
    elif mean < DARK_MEAN:
        issues.append("Film appears over-penetrated (too dark). Small nodules "
                      "and subtle lesions may be harder to see.")
        sev = "warn"
    elif mean > BRIGHT_MEAN_SEVERE:
        issues.append("Film is markedly under-penetrated (too bright). Lung "
                      "markings may strongly resemble infiltration.")
        sev = "severe"
    elif mean > BRIGHT_MEAN:
        issues.append("Film appears under-penetrated (too bright). Lung markings "
                      "may falsely resemble infiltration.")
        sev = "warn"
    if std < LOW_CONTRAST_STD and sev != "severe":
        issues.append("Low contrast — findings may be hard to distinguish.")
        sev = "warn"
    if dark_clip > DARK_CLIP_WARN or bright_clip > BRIGHT_CLIP_WARN:
        issues.append("Significant clipping — detail is lost in pure black or "
                      "pure white regions.")
        if sev == "pass":
            sev = "warn"

    return {"check": "exposure", "severity": sev,
            "value": {"mean": round(mean, 4), "std": round(std, 4),
                      "dark_clip": round(dark_clip, 4),
                      "bright_clip": round(bright_clip, 4)},
            "message": " ".join(issues) if issues
                       else "Exposure and contrast within usable range."}


def check_sharpness(a):
    """Motion blur / defocus. A blurred film hides exactly the small findings
    that matter most."""
    v = _laplacian_variance(a)
    if v < BLUR_SEVERE:
        sev, msg = "severe", ("Image appears significantly blurred — likely "
                              "patient movement. Small findings may be missed. "
                              "Repeat is advised.")
    elif v < BLUR_LAPLACIAN:
        sev, msg = "warn", "Image may be slightly blurred or soft."
    else:
        sev, msg = "pass", "Image sharpness adequate."
    return {"check": "sharpness", "severity": sev, "value": round(v, 2),
            "message": msg}


# findings most distorted by each technical fault — used to warn specifically
AFFECTED_BY = {
    "rotation":    ["Cardiomegaly", "Enlarged Cardiomediastinum", "Effusion",
                    "Atelectasis", "Pneumothorax"],
    "inspiration": ["Cardiomegaly", "Lung Opacity", "Consolidation",
                    "Infiltration", "Atelectasis", "Effusion"],
    "exposure":    ["Nodule", "Lung Lesion", "Mass", "Infiltration",
                    "Pneumothorax", "Emphysema"],
    "sharpness":   ["Nodule", "Lung Lesion", "Mass", "Fibrosis"],
}


def assess(img):
    """Run every check. Returns a dict the UI can render directly.

    Never raises: a failure in quality assessment must not prevent the
    radiograph from being read.
    """
    try:
        a = _to_gray_array(img)
        checks = [check_rotation(a), check_exposure(a), check_sharpness(a)]
        if ENABLE_INSPIRATION:
            checks.insert(1, check_inspiration(a))
    except Exception as e:
        return {"available": False, "error": str(e)[:120], "checks": [],
                "overall": "unknown", "warnings": [], "affected_findings": []}

    order = {"pass": 0, "warn": 1, "severe": 2, "unknown": 0}
    overall = max((c["severity"] for c in checks), key=lambda s: order.get(s, 0))

    warnings, affected = [], []
    for c in checks:
        if c["severity"] in ("warn", "severe"):
            warnings.append({"check": c["check"], "severity": c["severity"],
                             "message": c["message"]})
            for d in AFFECTED_BY.get(c["check"], []):
                if d not in affected:
                    affected.append(d)

    return {
        "available": True,
        "overall": overall,
        "checks": checks,
        "warnings": warnings,
        "affected_findings": affected,
        "summary": _summary(overall, warnings),
        "disclaimer": ("Heuristic technique assessment from image statistics, "
                       "not anatomical landmark measurement. Advisory only."),
    }


def _summary(overall, warnings):
    if overall == "pass":
        return "Film quality appears adequate for interpretation."
    if overall == "severe":
        return (f"Significant technical problem detected "
                f"({len(warnings)} issue(s)). Findings below may be unreliable "
                f"— consider repeating the radiograph.")
    return (f"Possible technical issue ({len(warnings)}). "
            f"Interpret the affected findings with caution.")


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, glob
    from pathlib import Path
    HERE = Path(__file__).resolve()
    ROOT = next((p for p in HERE.parents if (p / ".ayshcxr_root").exists()), None)
    if ROOT:
        os.chdir(ROOT)

    print("=" * 70)
    print("  image_quality self-test")
    print("=" * 70)

    # synthetic cases with known faults
    def show(title, img):
        r = assess(img)
        print(f"\n{title}")
        print(f"  overall: {r['overall'].upper()}  — {r['summary']}")
        for c in r["checks"]:
            print(f"    {c['check']:12s} {c['severity']:7s} {c['value']}")
        if r["affected_findings"]:
            print(f"    affected: {r['affected_findings'][:5]}")

    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]

    # crude synthetic 'chest': bright central mediastinum, dark lung fields
    base = np.full((h, w), 0.55, np.float32)
    base += 0.35 * np.exp(-((xx - w / 2) ** 2) / (2 * (w * 0.07) ** 2))   # spine
    for cx in (w * 0.28, w * 0.72):                                       # lungs
        base -= 0.40 * np.exp(-(((xx - cx) ** 2) / (2 * (w * 0.13) ** 2)
                                + ((yy - h * 0.45) ** 2) / (2 * (h * 0.20) ** 2)))
    base[int(h * 0.72):, :] += 0.30                                       # abdomen
    good = np.clip(base, 0, 1)
    show("GOOD film (synthetic)", Image.fromarray((good * 255).astype(np.uint8)))

    dark = np.clip(good * 0.35, 0, 1)
    show("OVER-PENETRATED (too dark)", Image.fromarray((dark * 255).astype(np.uint8)))

    bright = np.clip(good * 0.5 + 0.5, 0, 1)
    show("UNDER-PENETRATED (too bright)", Image.fromarray((bright * 255).astype(np.uint8)))

    rot = good.copy()
    rot[:, :w // 2] = np.clip(rot[:, :w // 2] + 0.22, 0, 1)   # one side denser
    show("ASYMMETRIC / rotated", Image.fromarray((rot * 255).astype(np.uint8)))

    k = np.ones((9, 9), np.float32) / 81.0
    blur = good.copy()
    for _ in range(6):                                          # cheap box blur
        blur = (blur + np.roll(blur, 1, 0) + np.roll(blur, -1, 0)
                + np.roll(blur, 1, 1) + np.roll(blur, -1, 1)) / 5.0
    show("BLURRED (motion)", Image.fromarray((blur * 255).astype(np.uint8)))

    # a real X-ray if one is around
    real = (glob.glob("data/vindr/train/*.png")[:1]
            or glob.glob("images_001/**/*.png", recursive=True)[:1])
    if real:
        show(f"REAL X-RAY  ({os.path.basename(real[0])})", Image.open(real[0]))

    # must never raise
    bad = assess("not an image")
    print(f"\nnon-image input -> available={bad['available']} (no exception) OK")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
