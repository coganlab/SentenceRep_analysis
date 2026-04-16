"""Supplementary Figure: Channel weight vs. behavioral coupling.

2x2 scatter grid showing normalised SliceTCA channel weight (x) vs.
Pearson r between each channel's HG time course and reaction time (y),
with linear fits and FDR-corrected statistics inset.

Two versions are saved:
  figure_S_w_vs_corr.{svg,png}            — filtered correlations
  figure_S_w_vs_corr_unfiltered.{svg,png}  — unfiltered correlations
"""
import os
import glob

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from mne.stats import fdr_correction

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE, DPI,
    COMP_NAMES, COMP_COLORS, LAYOUT, SM_PKL, SM_MODEL, EXCLUDE,
)
from analysis.fix.rt_hg_map import (
    normalize_subj_chan_name, get_sm_subgroup_w_values,
    _load_correlations_dict,
)

# ---------------------------------------------------------------------------
# Paths & layout
# ---------------------------------------------------------------------------
layout = LAYOUT
OUT_DERIV = os.path.join(layout.root, "derivatives", "rt_hg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SM_W_VALUE_THRESHOLD = 0.05
SM_SUBGROUPS = COMP_NAMES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_w_and_corr(filtered: bool = True):
    """Return per-component (w_vals, corr_vals) arrays with FDR-corrected p."""
    subgroup_w = get_sm_subgroup_w_values(
        layout, threshold=SM_W_VALUE_THRESHOLD,
        pkl_file=SM_PKL, model_file=SM_MODEL, subgroup_names=SM_SUBGROUPS,
    )
    all_corrs = _load_correlations_dict(OUT_DERIV, filtered=filtered,
                                        metric="r", significant_only=False)
    exclude_set = {normalize_subj_chan_name(n) for n in EXCLUDE}

    results = []
    for comp in SM_SUBGROUPS:
        w_vals, corr_vals = [], []
        for ch, w in subgroup_w[comp].items():
            if ch in exclude_set:
                continue
            if ch in all_corrs:
                w_vals.append(w)
                corr_vals.append(all_corrs[ch])
        if len(w_vals) >= 2:
            r, p = pearsonr(w_vals, corr_vals)
        else:
            r, p = np.nan, np.nan
        results.append({"comp": comp, "w": np.array(w_vals),
                        "corr": np.array(corr_vals), "r": r, "p": p})

    # FDR correction across the 4 tests
    raw_p = np.array([r["p"] for r in results])
    valid = np.isfinite(raw_p)
    fdr_p = np.full_like(raw_p, np.nan)
    if valid.any():
        _, fdr_p[valid] = fdr_correction(raw_p[valid])
    for i, res in enumerate(results):
        res["p_fdr"] = fdr_p[i]

    return results


# ---------------------------------------------------------------------------
# Build & save one version
# ---------------------------------------------------------------------------
def _build_figure(filtered: bool):
    tag = "" if filtered else "_unfiltered"
    filter_label = "filtered" if filtered else "unfiltered"

    fig = setup_figure(figsize=(12 * cm, 10 * cm))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    wc_data = _load_w_and_corr(filtered=filtered)

    for k, res in enumerate(wc_data):
        row, col = divmod(k, 2)
        ax = fig.add_subplot(gs[row, col])

        comp = res["comp"]
        w = res["w"]
        corr = res["corr"]

        ax.scatter(w, corr, s=8, alpha=0.5, c="grey", edgecolors="none")

        # Linear fit
        if len(w) >= 2 and np.isfinite(res["r"]):
            z = np.polyfit(w, corr, 1)
            xfit = np.linspace(w.min(), w.max(), 100)
            ax.plot(xfit, np.poly1d(z)(xfit), color="red", linewidth=1)

        # Stat text
        p_fdr = res["p_fdr"]
        sig = "*" if np.isfinite(p_fdr) and p_fdr < 0.05 else ""
        stat_txt = f"r={res['r']:.3f}\np={p_fdr:.3f}{sig}"
        ax.text(0.95, 0.95, stat_txt, transform=ax.transAxes,
                fontsize=LABEL_SIZE, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="grey", alpha=0.8))

        ax.set_title(comp, fontsize=LABEL_SIZE, color=COMP_COLORS[comp],
                     fontweight="bold")
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--", alpha=0.4)

        if row == 1:
            ax.set_xlabel("Channel weight", fontsize=LABEL_SIZE)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        if col == 0 and row == 0:
            ax.set_ylabel("HG\u2013RT corr (r)", fontsize=LABEL_SIZE)


    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, f"figure_S_w_vs_corr{tag}.svg"),
                bbox_inches="tight", dpi=DPI)
    fig.savefig(os.path.join(out_dir, f"figure_S_w_vs_corr{tag}.png"),
                bbox_inches="tight", dpi=DPI)
    print(f"Saved figure_S_w_vs_corr{tag}.svg / .png  ({filter_label})")
    plt.show()


# ---------------------------------------------------------------------------
# Generate both versions
# ---------------------------------------------------------------------------
_build_figure(filtered=True)
_build_figure(filtered=False)
