# make_figures.py
# AyShCXR — paper figures for AIDL-HCSY 2027 (IEEE, black-and-white safe).
# by Subhrakant Sethi & Ayush Singh
#
# Conference rules: vector graphics (EPS preferred), labels >= 4pt, readable in
# black-and-white print. So identity is carried by marker shape and line style,
# never by colour alone; everything is grey/black; one y-axis per panel.
#
#   python evaluation/make_figures.py --policy-cache results/quality_policy_records_site.csv
#   python evaluation/make_figures.py --real results/real_quality_summary_site.csv

import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_ROOT = next((p for p in _HERE.parents if (p / ".ayshcxr_root").exists()), None)
_os.chdir(_ROOT)
for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "evaluation")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "paper/figs"
COL_W = 3.45                     # IEEE single-column width, inches

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,      # embed TrueType: PDF compliance
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# policy -> (label, marker, fill) ; black/grey only
POLICY_STYLE = {
    "unc":      ("Uncertainty only",       "s", "white"),
    "qual_all": ("Quality, all calls",     "^", "white"),
    "qual_pos": ("Quality, positive calls", "o", "black"),
    "comb_all": ("Combined, all calls",    "v", "0.55"),
    "comb_pos": ("Combined, positive calls", "D", "0.55"),
}


def _save(fig, name):
    _os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT_DIR}/{name}.{ext}")
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300)      # preview only
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.pdf / .eps")


# ── Fig. 1: induced errors caught vs coverage (held-out half B) ───────────────
def fig_policies(cache, seed=42):
    from eval_quality_policies import add_decisions
    from threshold_overlap import threshold_tuning_films
    rec = pd.read_csv(cache)
    rec = add_decisions(rec[~rec["film"].isin(threshold_tuning_films())])
    films = np.array(sorted(rec["film"].unique()))
    perm = np.random.default_rng(seed).permutation(films)       # same split as analyze()
    half_b = set(perm[len(perm) // 2:])
    deg = rec[rec["film"].isin(half_b) & (rec["fault"] != "clean")]
    ind = deg["induced"].to_numpy()
    total = ind.sum()

    # uncertainty-ranked abstention swept over every coverage
    order = np.argsort(-deg["u"].to_numpy(), kind="stable")
    caught = np.concatenate([[0], np.cumsum(ind[order])]) / max(total, 1)
    cov = 1.0 - np.arange(len(order) + 1) / len(order)

    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    ax.plot(cov, caught, color="black", lw=1.0, label="Uncertainty-ranked (any coverage)")
    ax.plot([0, 1], [1, 0], color="0.45", lw=0.8, ls=(0, (3, 2)),
            label="Random abstention (expected)")
    for pol, (lab, mk, fc) in POLICY_STYLE.items():
        ab = deg["ab_" + pol].to_numpy()
        x, y = 1 - ab.mean(), ind[ab].sum() / max(total, 1)
        ax.plot([x], [y], ls="none", marker=mk, ms=5.5, mfc=fc, mec="black",
                mew=0.8, label=lab, zorder=3)
    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(-0.01, 0.75)
    ax.set_xlabel("Coverage (fraction of findings still reported)")
    ax.set_ylabel("Share of induced errors caught")
    # legend below the axes: inside, it collides with the policy markers
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2,
              frameon=False, handlelength=2.0, columnspacing=1.0)
    _save(fig, "fig_policies")


# ── Fig. 2: real-film AUC drop by check and finding ──────────────────────────
def fig_real(summary, view="PA"):
    r = pd.read_csv(summary)
    r = r[(r["view"] == view)].dropna(subset=["auc_drop"])
    checks = [c for c in ["rotation", "exposure", "sharpness"] if c in set(r["check"])]
    if not checks:
        print("  no real-film rows to plot")
        return
    fig, axes = plt.subplots(1, len(checks), figsize=(COL_W * 2.05, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    findings = sorted(r["finding"].unique())
    ypos = {f: i for i, f in enumerate(findings)}
    for ax, c in zip(axes, checks):
        s = r[r["check"] == c]
        for _, row in s.iterrows():
            y = ypos[row["finding"]]
            if np.isfinite(row["drop_ci_lo"]):
                ax.plot([row["drop_ci_lo"], row["drop_ci_hi"]], [y, y],
                        color="0.35", lw=0.8)
            aff = bool(row["in_affected_by"])
            ax.plot([row["auc_drop"]], [y], marker="o" if aff else "s", ms=4.5,
                    mfc="black" if aff else "white", mec="black", mew=0.7, ls="none")
        ax.axvline(0, color="0.3", lw=0.6)
        ax.set_title(f"{c.capitalize()} flagged (n={int(s['n_flagged'].iloc[0])})",
                     fontsize=7.5)
        ax.set_xlabel("AUC drop (clean $-$ flagged)")
    axes[0].set_yticks(range(len(findings)))
    axes[0].set_yticklabels(findings)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", mfc="black", mec="black", ls="none",
                      label="Finding the check is mapped to"),
               Line2D([], [], marker="s", mfc="white", mec="black", ls="none",
                      label="Finding not mapped"),
               Line2D([], [], color="0.35", lw=0.8, label="95% CI (patient bootstrap)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    _save(fig, f"fig_real_{view.lower()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-cache", default="")
    ap.add_argument("--real", default="")
    a = ap.parse_args()
    if a.policy_cache:
        fig_policies(a.policy_cache)
    if a.real:
        for v in ("PA", "AP"):
            fig_real(a.real, v)


if __name__ == "__main__":
    main()
