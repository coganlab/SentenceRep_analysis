"""Supplementary Figure: SliceTCA component time-courses by condition (LS / LM / JL).

For each of the four sliceTCA components, plots the H temporal factor across
the three condition pairs (LS solid, LM dashed, JL dotted) for both the
stimulus- and go-cue-aligned epochs.  Key purpose: shows that the WM
component carries residual delay-period activity in the Just-Listen
condition (Discussion paragraph on implicit phonological rehearsal).

Adapted from the inline plot at
``analysis/decomposition/slice_freq.py:553-603``.
"""
import os
import pickle
from functools import partial

import matplotlib.gridspec as gridspec
import numpy as np
import torch
import slicetca

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE,
    COMP_NAMES, COMP_COLORS_LIST,
    LAYOUT, SM_PKL, SM_MODEL,
    XLABEL_STIMULUS, XLABEL_GO,
    finalize_figure, add_panel_label,
)
from analysis.load import load_spec, split_and_stack
from ieeg.viz.ensemble import plot_dist

# ---------------------------------------------------------------------------
# Load model & data (mirrors figure_4)
# ---------------------------------------------------------------------------
layout = LAYOUT
group = "SM"
folder = "stats_freq_hilbert"
conds_ordered = ['aud_ls', 'go_ls', 'aud_lm', 'go_lm', 'aud_jl', 'go_jl']

with open(SM_PKL, "rb") as f:
    labels = pickle.load(f)

state = torch.load(SM_MODEL, map_location="cpu")
shape = state["vectors.0.0"].shape[1:] + state["vectors.0.1"].shape[1:]
n_components = state["vectors.0.0"].shape[0]
model = slicetca.core.SliceTCA(
    dimensions=shape, ranks=(n_components, 0, 0, 0), positive=True,
    initialization="uniform-positive", dtype=torch.float32, lr=5e-4,
    weight_decay=partial(torch.optim.Adam, eps=1e-9),
    loss=torch.nn.L1Loss(reduction="mean"),
    init_bias=0.1, threshold=None, patience=None,
)
model.load_state_dict(state)
W, H = model.get_components(numpy=True)[0]
# H shape: (n_components, 3 cond_pairs, freqs, time=350)

# Time slices (mirrors slice_freq.py:553-558 — aud and go are the two halves
# of the 350-sample concatenated input)
TIMINGS = {"aud": slice(0, 175), "go": slice(175, 350)}
TIME_RANGE = (-0.5, 1.25)  # both aud and go windows span this
COND_PAIR_LABELS = ["LS", "LM", "JL"]
COND_PAIR_LINESTYLES = ["-", "--", ":"]

# ---------------------------------------------------------------------------
# Figure: rows = components, cols = (aud, go)
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(14 * cm, 18 * cm))
gs = gridspec.GridSpec(n_components, 2, figure=fig,
                       hspace=0.3, wspace=0.1)

ymin = +np.inf
ymax = -np.inf
axes = {}

# First pass: plot and collect ylims
for c in range(n_components):
    for j, (epoch, sl) in enumerate(TIMINGS.items()):
        ax = fig.add_subplot(gs[c, j])
        axes[(c, j)] = ax
        for k in range(3):  # LS, LM, JL
            data = H[c, k, ..., sl]  # (freqs, time)
            plot_dist(data, ax=ax, color=COMP_COLORS_LIST[c],
                      mode="sem", times=TIME_RANGE,
                      linestyle=COND_PAIR_LINESTYLES[k],
                      label=COND_PAIR_LABELS[k])
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        # Component label on left
        if j == 0:
            ax.set_ylabel(f"{COMP_NAMES[c]}\n(z-HG)", fontsize=LABEL_SIZE,
                          color=COMP_COLORS_LIST[c])
        else:
            ax.set_yticklabels([])
        # Hide x-tick labels for non-bottom rows
        if c < n_components - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(XLABEL_STIMULUS if epoch == "aud" else XLABEL_GO,
                          fontsize=LABEL_SIZE)
        # Legend on top-right cell only
        if c == 0 and j == 1:
            ax.legend(fontsize=TICK_SIZE, loc="upper right",
                      framealpha=0.7)
        yl = ax.get_ylim()
        ymin = min(ymin, yl[0])
        ymax = max(ymax, yl[1])

# Unify y-limits within row (each component on its own scale is more
# informative than a global scale across components)
for c in range(n_components):
    row_ymin = +np.inf
    row_ymax = -np.inf
    for j in range(2):
        yl = axes[(c, j)].get_ylim()
        row_ymin = min(row_ymin, yl[0])
        row_ymax = max(row_ymax, yl[1])
    for j in range(2):
        axes[(c, j)].set_ylim(row_ymin, row_ymax)
        # 0-line marker
        axes[(c, j)].axhline(0, color='k', linewidth=0.4,
                             linestyle='--', alpha=0.3)

add_panel_label(axes[(0, 0)], "a", x=-0.1, y=1.05)

finalize_figure(fig, "figure_S_jl_components")
