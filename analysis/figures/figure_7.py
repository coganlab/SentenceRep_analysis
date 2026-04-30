"""Figure 7: Behavioral grounding of the component decomposition.

Single panel:
  Per-component ridge-regression R² between component-weighted HG and
  reaction time, with permutation-cluster significance bars.
"""
import os
import pickle
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import slicetca

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE, DPI,
    COMP_NAMES, COMP_COLORS_LIST,
    LAYOUT, SM_PKL, SM_MODEL, EXCLUDE,
    XLABEL_GO,
)
from analysis.grouping import group_elecs
from analysis.load import load_data, exclude
from analysis.utils import load_response_times
from analysis.utils.plotting import plot_horizontal_bars
from analysis.decoding.rt_prediction_comp import (
    get_unique_subjects, make_rt_array, weighted_preserve_stats,
)
from ieeg.calc.stats import time_perm_cluster, ridge_nd

# ---------------------------------------------------------------------------
# Paths & data
# ---------------------------------------------------------------------------
layout = LAYOUT

# Text size
FS = LABEL_SIZE
SM_SUBGROUPS = COMP_NAMES


# ===================================================================
# RT prediction (mirrors rt_prediction_comp.py)
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
def _build_figure():
    fig = setup_figure(figsize=(10 * cm, 8 * cm))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax_b = fig.add_subplot(gs[0, 0])

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

    ax_b.set_xlabel(XLABEL_GO, fontsize=FS)
    ax_b.set_ylabel("R\u00b2", fontsize=FS)
    ax_b.set_title("RT prediction (go_ls)", fontsize=FS)
    ax_b.tick_params(labelsize=FS)
    ax_b.legend(fontsize=FS, loc="upper left", framealpha=0.6)

    # ----------------------------- Save ------------------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, "figure_7.svg"),
                bbox_inches="tight", dpi=DPI)
    fig.savefig(os.path.join(out_dir, "figure_7.png"),
                bbox_inches="tight", dpi=DPI)
    print("Saved figure_7.svg / .png")
    plt.show()


# ---------------------------------------------------------------------------
_build_figure()
