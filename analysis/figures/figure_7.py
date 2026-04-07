"""Figure 7: Behavioral grounding of the component decomposition.

Layout (1 x 2):
  (a)  HG peak timing by component — box plots of mean HG response peak
       times (per electrode, averaged across trials) for Auditory, WM,
       Motor, and Visual SliceTCA component groups, with individual
       electrode peaks overlaid as grey dots.  ANOVA and FDR-corrected
       pairwise comparisons annotated.
  (b)  Channel weight vs. behavioral coupling — 2x2 scatter grid showing
       normalised SliceTCA channel weight (x) vs. Pearson r between each
       channel's HG time course and reaction time (y), with linear fits
       and FDR-corrected statistics inset.
"""
import os
import glob

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, f_oneway
from mne.stats import fdr_correction

from analysis.figures.config import cm, GS_KWARGS, setup_figure, LABEL_SIZE, TICK_SIZE
from analysis.fix.rt_hg_map import (
    normalize_subj_chan_name, get_sm_subgroup_mapping,
    get_sm_subgroup_w_values, filter_peak_times_by_rt,
    _load_correlations_dict, pairwise_comparisons,
)
from ieeg.io import get_data

# ---------------------------------------------------------------------------
# Paths & layout
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
if "SLURM_ARRAY_TASK_ID" in os.environ:
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
else:
    LAB_root = os.path.join(HOME, "Box", "CoganLab")

layout = get_data("SentenceRep", root=LAB_root)
OUT_DERIV = os.path.join(layout.root, "derivatives", "rt_hg")

# ---------------------------------------------------------------------------
# Constants (match rt_hg_map.py main section)
# ---------------------------------------------------------------------------
SM_MAPPING_THRESHOLD = 0.4
SM_W_VALUE_THRESHOLD = 0.05
SM_SUBGROUPS = ["Auditory", "WM", "Motor", "Visual"]
COMP_COLORS = {"Auditory": "tab:blue", "WM": "tab:green",
               "Motor": "tab:red", "Visual": "goldenrod"}

EXCLUDE = [
    "D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
    "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
    "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
    "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
    "D0027-LPIF10", "D0026-RPG20", "D0026-RPG21", "D0026-RPG28",
    "D0026-RPG29", "D0026-RPG36", "D0007-RFG44",
]

# Paths for SliceTCA model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
DECODING_DIR = os.path.join(ANALYSIS_DIR, "decoding")
SM_PKL = os.path.join(DECODING_DIR, "SM_chns.pkl")
SM_MODEL = os.path.join(DECODING_DIR, "model_SM2_freq.pt")

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_mean_peak_times_by_component():
    """Load NPZ files, average peak times per electrode, group by component.

    Returns dict[component_name] -> 1-D array of mean peak times (one per
    electrode).
    """
    sm_mapping = get_sm_subgroup_mapping(
        layout, threshold=SM_MAPPING_THRESHOLD,
        pkl_file=SM_PKL, model_file=SM_MODEL, subgroup_names=SM_SUBGROUPS,
    )
    exclude_set = {normalize_subj_chan_name(n) for n in EXCLUDE}

    npz_files = glob.glob(os.path.join(OUT_DERIV, "*_rt_hg_peak.npz"))
    # Accumulate per-electrode lists keyed by (subj, channel, component)
    elec_peaks = {g: {} for g in SM_SUBGROUPS}

    for npz_path in npz_files:
        subj = os.path.basename(npz_path).replace("_rt_hg_peak.npz", "")
        data = np.load(npz_path, allow_pickle=True)
        peak_times = data["peak_time"]       # (trials, channels)
        rt_vec = data["response_time"]       # (trials,)
        ch_names = data["channel_names"]     # (channels,)

        # Filter peaks that exceed response time
        peak_times = filter_peak_times_by_rt(peak_times, rt_vec)

        for ch_idx, ch in enumerate(ch_names):
            norm = normalize_subj_chan_name(f"{subj}-{ch}")
            if norm in exclude_set:
                continue
            groups = sm_mapping.get(norm, [])
            col = peak_times[:, ch_idx]
            valid = col[np.isfinite(col)]
            if len(valid) == 0:
                continue
            mean_pt = np.nanmean(valid)
            for g in groups:
                if g in elec_peaks:
                    elec_peaks[g][norm] = mean_pt

    return {g: np.array(list(v.values())) for g, v in elec_peaks.items()
            if len(v) > 0}


def _load_w_and_corr():
    """Return per-component (w_vals, corr_vals) arrays for panel (b).

    Also returns FDR-corrected p-values for each component regression.
    """
    subgroup_w = get_sm_subgroup_w_values(
        layout, threshold=SM_W_VALUE_THRESHOLD,
        pkl_file=SM_PKL, model_file=SM_MODEL, subgroup_names=SM_SUBGROUPS,
    )
    all_corrs = _load_correlations_dict(OUT_DERIV, filtered=True, metric="r",
                                        significant_only=False)
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
# Build the figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(18 * cm, 9 * cm))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6],
                       wspace=0.35)

# ========================= Panel (a): box plots ============================
ax_a = fig.add_subplot(gs[0, 0])

peak_data = _load_mean_peak_times_by_component()
groups_ordered = [g for g in SM_SUBGROUPS if g in peak_data]
box_arrays = [peak_data[g] for g in groups_ordered]

bp = ax_a.boxplot(
    box_arrays, tick_labels=groups_ordered, patch_artist=True,
    widths=0.55, showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="white",
                   markeredgecolor="black", markersize=3),
    medianprops=dict(color="k", linewidth=1),
    flierprops=dict(marker="", linewidth=0),  # hide outlier markers
)
for patch, grp in zip(bp["boxes"], groups_ordered):
    patch.set_facecolor(COMP_COLORS[grp])
    patch.set_alpha(0.55)

# Overlay individual electrode dots
for i, grp in enumerate(groups_ordered):
    vals = peak_data[grp]
    jitter = np.random.default_rng(42).normal(0, 0.06, size=len(vals))
    ax_a.scatter(i + 1 + jitter, vals, s=6, c="grey", alpha=0.45,
                 edgecolors="none", zorder=5)

# ANOVA
f_stat, p_anova = f_oneway(*box_arrays)

# Pairwise comparisons (uncorrected first, so we can pool with ANOVA for joint FDR)
pw = pairwise_comparisons(
    {g: peak_data[g] for g in groups_ordered},
    correction_method="none",  # no internal correction
)

# Joint FDR correction: pool ANOVA p-value with all pairwise p-values
all_p = np.concatenate([[p_anova], pw["pvalue"].values])
valid = np.isfinite(all_p)
all_p_fdr = np.full_like(all_p, np.nan)
if valid.any():
    _, all_p_fdr[valid] = fdr_correction(all_p[valid])

p_anova_fdr = all_p_fdr[0]
pw["pvalue_corrected"] = all_p_fdr[1:]
pw["significant"] = pw["pvalue_corrected"] < 0.05

ax_a.set_title(f"ANOVA: F={f_stat:.2f}, p={p_anova_fdr:.2e} (FDR)",
               fontsize=TICK_SIZE)

# Pairwise brackets (row-wrapped layout matching rt_hg_map)
y_max = ax_a.get_ylim()[1]
bracket_height = y_max * 0.05
y_start = y_max * 0.9
bracket_rows = 3
x_positions = {g: i + 1 for i, g in enumerate(groups_ordered)}

for idx, (_, row) in enumerate(pw.iterrows()):
    g1, g2 = row["group1"], row["group2"]
    if g1 not in x_positions or g2 not in x_positions:
        continue
    x1, x2 = x_positions[g1], x_positions[g2]
    y = y_start + (idx % bracket_rows) * bracket_height * 1.5
    color = "red" if row["significant"] else "black"
    lw = 1.0 if row["significant"] else 0.6
    ax_a.plot([x1, x1, x2, x2],
              [y, y + bracket_height, y + bracket_height, y],
              color=color, linewidth=lw, clip_on=False)
    p_text = f"p={row['pvalue_corrected']:.3f}"
    ax_a.text((x1 + x2) / 2, y + bracket_height * 1.2, p_text,
              ha="center", va="bottom", fontsize=TICK_SIZE - 1, color=color)

ax_a.set_ylabel("HG peak time (s post-go)", fontsize=LABEL_SIZE)
ax_a.set_xlabel("Component", fontsize=LABEL_SIZE)
ax_a.tick_params(labelsize=TICK_SIZE)
ax_a.grid(True, alpha=0.3, axis='y')

# Add sample sizes and mean/std as text boxes
for i, grp in enumerate(groups_ordered):
    n = len(peak_data[grp])
    mean_val = peak_data[grp].mean()
    std_val = peak_data[grp].std()
    ax_a.text(i + 1, y_max * 0.95, f"n={n}\n\u03bc={mean_val:.3f}\n\u03c3={std_val:.3f}",
              ha="center", va="top", fontsize=TICK_SIZE - 1,
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

# Panel label
fig.text(0.01, 0.97, "a", fontsize=LABEL_SIZE + 2, fontweight="bold",
         va="top")

# ========================= Panel (b): W vs corr ===========================
gs_b = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1],
                                        hspace=0.45, wspace=0.35)
wc_data = _load_w_and_corr()

fig.text(0.44, 0.97, "b", fontsize=LABEL_SIZE + 2, fontweight="bold",
         va="top")

for k, res in enumerate(wc_data):
    row_b, col_b = divmod(k, 2)
    ax = fig.add_subplot(gs_b[row_b, col_b])

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
    ax.text(0.95, 0.95, stat_txt, transform=ax.transAxes, fontsize=TICK_SIZE,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    ax.set_title(comp, fontsize=LABEL_SIZE, color=COMP_COLORS[comp],
                 fontweight="bold")
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--", alpha=0.4)

    if row_b == 1:
        ax.set_xlabel("Channel weight", fontsize=LABEL_SIZE)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)

    if col_b == 0:
        ax.set_ylabel("HG–RT corr (r)", fontsize=LABEL_SIZE)

    ax.tick_params(labelsize=TICK_SIZE)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_7.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_7.png"), bbox_inches="tight",
            dpi=300)
plt.show()
