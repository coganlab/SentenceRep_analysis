"""Figure 7: Behavioral grounding of the component decomposition.

Layout (1 x 2):
  (a)  HG peak timing by component — box plots of mean HG response peak
       times (per electrode, averaged across trials) for Auditory, WM,
       Motor, and Visual SliceTCA component groups, with individual
       electrode peaks overlaid as grey dots.  ANOVA and FDR-corrected
       pairwise comparisons annotated.  A thin dotted line marks the
       grand-average reaction time.
  (b)  RT prediction R² over time — per-component ridge-regression R²
       between component-weighted HG and reaction time, with
       permutation-cluster significance bars.

Two versions are saved:
  figure_7.{svg,png}            — filtered (peak_time > response_time -> NaN)
  figure_7_unfiltered.{svg,png} — all trials kept regardless of peak vs RT
"""
import os
import glob
import pickle
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import slicetca
from scipy.stats import f_oneway

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE,
    COMP_NAMES, COMP_COLORS, COMP_COLORS_LIST,
    LAYOUT, SM_PKL, SM_MODEL, EXCLUDE, DECODING_DIR,
)
from analysis.fix.rt_hg_map import (
    normalize_subj_chan_name, get_sm_subgroup_mapping,
    filter_peak_times_by_rt, pairwise_comparisons,
)
from analysis.grouping import group_elecs
from analysis.load import load_data, load_spec, exclude
from analysis.utils import load_response_times
from analysis.utils.plotting import plot_horizontal_bars
from analysis.decoding.rt_prediction_comp import (
    get_unique_subjects, make_rt_array, weighted_preserve_stats,
)
from ieeg.arrays.label import LabeledArray
from ieeg.calc.stats import time_perm_cluster, ridge_nd

# ---------------------------------------------------------------------------
# Paths & data
# ---------------------------------------------------------------------------
layout = LAYOUT
OUT_DERIV = os.path.join(layout.root, "derivatives", "rt_hg")

# ---------------------------------------------------------------------------
# Constants — colours match figure_4 / figure_5 / figure_6
# ---------------------------------------------------------------------------
SM_MAPPING_THRESHOLD = 0.4
SM_SUBGROUPS = COMP_NAMES

# Text size — everything at LABEL_SIZE (7 pt)
FS = LABEL_SIZE


# ===================================================================
# Panel (a) helpers
# ===================================================================
def _load_mean_peak_times_and_rt(filtered: bool = True):
    """Load NPZ files, average peak times per electrode, group by component.

    Returns
    -------
    peak_dict : dict[str, np.ndarray]
    mean_rt : float
    """
    sm_mapping = get_sm_subgroup_mapping(
        layout, threshold=SM_MAPPING_THRESHOLD,
        pkl_file=SM_PKL, model_file=SM_MODEL, subgroup_names=SM_SUBGROUPS,
    )
    exclude_set = {normalize_subj_chan_name(n) for n in EXCLUDE}

    npz_files = glob.glob(os.path.join(OUT_DERIV, "*_rt_hg_peak.npz"))
    elec_peaks = {g: {} for g in SM_SUBGROUPS}
    all_rts = []

    for npz_path in npz_files:
        subj = os.path.basename(npz_path).replace("_rt_hg_peak.npz", "")
        data = np.load(npz_path, allow_pickle=True)
        peak_times = data["peak_time"]
        rt_vec = data["response_time"]
        ch_names = data["channel_names"]

        valid_rt = rt_vec[np.isfinite(rt_vec)]
        if len(valid_rt) > 0:
            all_rts.append(valid_rt)

        if filtered:
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

    peak_dict = {g: np.array(list(v.values())) for g, v in elec_peaks.items()
                 if len(v) > 0}
    mean_rt = float(np.nanmean(np.concatenate(all_rts))) if all_rts else np.nan
    return peak_dict, mean_rt


# ===================================================================
# Panel (b) helpers — RT prediction (mirrors rt_prediction_comp.py)
# ===================================================================
def _compute_rt_prediction():
    """Run per-component ridge RT prediction for go_ls condition.

    Returns
    -------
    corrs : np.ndarray (n_components, n_time)
    sig_masks : np.ndarray (n_components, n_time)  boolean
    timing : tuple (t_start, t_end)
    """
    group = "SM"
    folder = "stats_freq_hilbert"
    cond = "go_ls"
    timing = (-0.5, 1.5)
    n_permutations = 1000

    # Load grouping
    sigs = load_data(layout, folder, "mask")
    AUD, SM, PROD, sig_chans, delay = group_elecs(
        sigs, [s for s in sigs.labels[1] if s not in exclude], sigs.labels[0])
    zscores = load_data(layout, folder, "zscore")

    # Load SliceTCA model
    labels = pickle.load(open(SM_PKL, "rb"))
    state = torch.load(SM_MODEL, map_location="cpu")
    shape = state["vectors.0.0"].shape[1:] + state["vectors.0.1"].shape[1:]
    n_comp = state["vectors.0.0"].shape[0]
    model = slicetca.core.SliceTCA(
        dimensions=shape,
        ranks=(n_comp, 0, 0, 0),
        positive=True,
        initialization="uniform-positive",
        dtype=torch.float32,
        lr=5e-4,
        weight_decay=partial(torch.optim.Adam, eps=1e-9),
        loss=torch.nn.L1Loss(reduction="mean"),
        init_bias=0.1,
        threshold=None,
        patience=None,
    )
    model.load_state_dict(state)
    W, H = model.get_components(numpy=True)[0]

    # Load response times
    subjects_list_rt = list(
        set(s.split("-")[0] for s in sigs.labels[1] if s not in exclude))
    rt_dict, trial_df_dict = load_response_times(layout, subjects_list_rt)

    corrs_all = []
    sigs_all = []

    for i in range(n_comp):
        subset = np.nonzero(W[i] / W.mean() > 0.05)[0]
        comp_chans = [labels[0][s] for s in subset]
        new_labels = [comp_chans, labels[2]]
        in_data_full = (zscores[cond][:, comp_chans, ..., :175]
                        .combine((0, 3)).dropna())

        weights = model.construct_single_component(0, i).detach().numpy()[subset]
        weights_go = weights[:, 0, :, None, 175:]

        unique_subjects = get_unique_subjects(comp_chans, zscores.labels[2])
        rt_array = make_rt_array(rt_dict, trial_df_dict, new_labels,
                                 unique_subjects, exclude)

        weighted_preserve_stats(in_data_full, weights_go)
        in_data = np.nanmean(in_data_full, axis=1)

        y = np.array(rt_array)
        valid_idx = ~np.isnan(in_data[..., 0]) & ~np.isnan(y)
        x = np.array(in_data, dtype="f2")[valid_idx]
        y_broad = np.broadcast_to(
            np.array(rt_array, dtype="f2")[valid_idx, None], x.shape)

        results = time_perm_cluster(
            x, y_broad, 0.05,
            stat_func=ridge_nd,
            permutation_type="pairings",
            vectorized=True,
            n_perm=n_permutations,
            tails=1,
            axis=0,
        )

        corr, _, _ = ridge_nd(x, y_broad, 0, return_params=True)
        corrs_all.append(corr)
        sigs_all.append(results[0])

    return np.array(corrs_all), np.array(sigs_all), timing, n_comp


# ===================================================================
# Build figure
# ===================================================================
def _build_figure(filtered: bool):
    tag = "" if filtered else "_unfiltered"
    filter_label = "filtered" if filtered else "unfiltered"

    fig = setup_figure(figsize=(18 * cm, 8 * cm))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3],
                           wspace=0.35)

    # ======================= Panel (a): box plots ==========================
    ax_a = fig.add_subplot(gs[0, 0])

    peak_data, mean_rt = _load_mean_peak_times_and_rt(filtered=filtered)
    groups_ordered = [g for g in SM_SUBGROUPS if g in peak_data]
    box_arrays = [peak_data[g] for g in groups_ordered]

    bp = ax_a.boxplot(
        box_arrays, tick_labels=groups_ordered, patch_artist=True,
        widths=0.55, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=3),
        medianprops=dict(color="k", linewidth=1),
        flierprops=dict(marker="", linewidth=0),
    )
    for patch, grp in zip(bp["boxes"], groups_ordered):
        patch.set_facecolor(COMP_COLORS[grp])
        patch.set_alpha(0.55)

    # Individual electrode dots
    for i, grp in enumerate(groups_ordered):
        vals = peak_data[grp]
        jitter = np.random.default_rng(42).normal(0, 0.06, size=len(vals))
        ax_a.scatter(i + 1 + jitter, vals, s=6, c="grey", alpha=0.45,
                     edgecolors="none", zorder=5)

    # Mean RT reference line
    if np.isfinite(mean_rt):
        ax_a.axhline(mean_rt, color="k", linewidth=0.8, linestyle=":",
                     alpha=0.7, zorder=4)
        ax_a.text(len(groups_ordered) + 0.45, mean_rt,
                  f"mean RT = {mean_rt:.3f} s", va="center", ha="right",
                  fontsize=FS, color="k", style="italic")

    # ANOVA
    f_stat, p_anova = f_oneway(*box_arrays)
    ax_a.set_title(
        f"ANOVA: F={f_stat:.2f}, p={p_anova:.2e}\n({filter_label})",
        fontsize=FS)

    # Pairwise brackets
    pw = pairwise_comparisons(
        {g: peak_data[g] for g in groups_ordered},
        correction_method="fdr_bh",
    )
    y_top = max(v.max() for v in box_arrays)
    bracket_y = y_top * 1.05
    step = y_top * 0.08

    for idx, (_, row) in enumerate(pw.iterrows()):
        g1, g2 = row["group1"], row["group2"]
        if g1 not in groups_ordered or g2 not in groups_ordered:
            continue
        x1 = groups_ordered.index(g1) + 1
        x2 = groups_ordered.index(g2) + 1
        y = bracket_y + idx * step
        color = "red" if row["significant"] else "black"
        lw = 1.0 if row["significant"] else 0.6
        ax_a.plot([x1, x1, x2, x2],
                  [y, y + step * 0.3, y + step * 0.3, y],
                  color=color, linewidth=lw, clip_on=False)
        p_text = f"p={row['pvalue_corrected']:.3f}"
        ax_a.text((x1 + x2) / 2, y + step * 0.35, p_text, ha="center",
                  va="bottom", fontsize=FS, color=color)

    ax_a.set_ylabel("HG peak time (s post-go)", fontsize=FS)
    ax_a.set_xlabel("Component", fontsize=FS)
    ax_a.tick_params(labelsize=FS)
    ax_a.grid(True, alpha=0.3, axis="y")

    # Sample sizes and mean/std
    for i, grp in enumerate(groups_ordered):
        n = len(peak_data[grp])
        mean_val = peak_data[grp].mean()
        std_val = peak_data[grp].std()
        ax_a.text(i + 1, ax_a.get_ylim()[1] * 0.95,
                  f"n={n}\n\u03bc={mean_val:.3f}\n\u03c3={std_val:.3f}",
                  ha="center", va="top", fontsize=FS,
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.text(0.01, 0.97, "a", fontsize=FS + 2, fontweight="bold", va="top")

    # ======================= Panel (b): RT prediction ======================
    ax_b = fig.add_subplot(gs[0, 1])

    corrs, sig_masks, timing, n_comp = _compute_rt_prediction()
    time_axis = np.linspace(timing[0], timing[1], corrs.shape[-1])

    for i in range(n_comp):
        ax_b.plot(time_axis, corrs[i], label=SM_SUBGROUPS[i],
                  color=COMP_COLORS_LIST[i], linewidth=1)

    ax_b.set_ylim(bottom=-0.01)
    ax_b.axhline(0, color="k", linestyle="--", linewidth=0.5)

    # Significance bars
    bar_times = [(timing[0], timing[1])] * n_comp
    plot_horizontal_bars(ax_b, list(sig_masks), colors=COMP_COLORS_LIST,
                         times=bar_times)

    ax_b.set_xlabel("Time from go cue (s)", fontsize=FS)
    ax_b.set_ylabel("R\u00b2", fontsize=FS)
    ax_b.set_title("RT prediction (go_ls)", fontsize=FS)
    ax_b.tick_params(labelsize=FS)
    ax_b.legend(fontsize=FS, loc="upper left", framealpha=0.6)

    fig.text(0.47, 0.97, "b", fontsize=FS + 2, fontweight="bold", va="top")

    # ----------------------------- Save ------------------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, f"figure_7{tag}.svg"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"figure_7{tag}.png"),
                bbox_inches="tight", dpi=300)
    print(f"Saved figure_7{tag}.svg / .png  ({filter_label})")
    plt.show()


# ---------------------------------------------------------------------------
# Generate both versions
# ---------------------------------------------------------------------------
_build_figure(filtered=True)
_build_figure(filtered=False)
