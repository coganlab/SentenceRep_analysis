"""Supplementary Figure: AUD / SM / PROD HG by condition (LS / LM / JL).

Per-group analogue of ``figure_S_jl_components.py``: for each of the three
functional groupings we plot the frequency-averaged HG response across the
three condition pairs (LS solid, LM dashed, JL dotted), separately for the
stimulus- and go-cue-aligned epochs.

Same data path as figure 2 (``derivatives/stats_freq_hilbert/combined/zscore``).
"""
import os
import sys

import matplotlib.gridspec as gridspec
import numpy as np

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE, LAYOUT,
    XLABEL_STIMULUS, XLABEL_GO,
    finalize_figure, add_panel_label,
)
from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray
from ieeg.viz.ensemble import plot_dist

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
folder = "stats_freq_hilbert"
derivatives = os.path.join(LAYOUT.root, "derivatives", folder, "combined")

print("Loading mask...", flush=True)
sigs = LabeledArray.fromfile(os.path.join(derivatives, "mask"))
AUD, SM, PROD, *_ = group_elecs(sigs, sigs.labels[1], sigs.labels[0])
print(f"Groups: AUD={len(AUD)}, SM={len(SM)}, PROD={len(PROD)}", flush=True)

print("Loading zscores (mmap)...", flush=True)
zscores = LabeledArray.fromfile(os.path.join(derivatives, "zscore"),
                                mmap_mode="r")
print(f"zscores conds: {zscores.labels[0]}", flush=True)


def freq_avg(cond: str) -> np.ndarray:
    raw = zscores[cond].__array__()
    if raw.ndim == 3:
        return np.nanmean(raw, axis=1)
    if raw.ndim == 4:
        return np.nanmean(raw, axis=(0, 2))
    keep = {1, raw.ndim - 1}
    return np.nanmean(raw, axis=tuple(i for i in range(raw.ndim) if i not in keep))


GROUPS = [
    ("Auditory", AUD, "green"),
    ("Sensory-Motor", SM, "red"),
    ("Production", PROD, "blue"),
]

EPOCHS = [
    ("aud", ("aud_ls", "aud_lm", "aud_jl"), (-0.5, 1.5), XLABEL_STIMULUS),
    ("go", ("go_ls", "go_lm", "go_jl"), (-0.5, 1.5), XLABEL_GO),
]

# Pre-load every condition once (loop above re-loads each per-group×cond
# combination, which is expensive for mmap'd Box-stored data).
print("Pre-loading conditions...", flush=True)
cond_cache = {}
for _, conds, _, _ in EPOCHS:
    for c in conds:
        if c in zscores.labels[0] and c not in cond_cache:
            print(f"  loading {c}...", flush=True)
            cond_cache[c] = freq_avg(c)
            print(f"    {cond_cache[c].shape}", flush=True)
print("Pre-load done.", flush=True)

LINESTYLES = ["-", "--", ":"]
COND_LABELS = ["LS", "LM", "JL"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(14 * cm, 14 * cm))
gs = gridspec.GridSpec(len(GROUPS), 2, figure=fig,
                       hspace=0.3, wspace=0.1)

axes = {}
for r, (gname, idx, color) in enumerate(GROUPS):
    if not idx:
        continue
    idx_list = sorted(idx)
    for c, (epoch_key, conds, trange, xlabel) in enumerate(EPOCHS):
        ax = fig.add_subplot(gs[r, c])
        axes[(r, c)] = ax
        for k, cond in enumerate(conds):
            if cond in cond_cache:
                arr = cond_cache[cond][idx_list]
                plot_dist(arr, times=trange, color=color,
                          mode="sem", linestyle=LINESTYLES[k],
                          label=COND_LABELS[k], ax=ax)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        if c == 0:
            ax.set_ylabel(f"{gname}\n(z-HG)", fontsize=LABEL_SIZE,
                          color=color)
        else:
            ax.set_yticklabels([])
        if r < len(GROUPS) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        if r == 0 and c == 1:
            ax.legend(fontsize=TICK_SIZE, loc="upper right",
                      framealpha=0.7)

# Unify y-limits within each row
for r in range(len(GROUPS)):
    ymin, ymax = float('inf'), float('-inf')
    for c in range(2):
        ax = axes.get((r, c))
        if ax is None:
            continue
        yl = ax.get_ylim()
        ymin = min(ymin, yl[0])
        ymax = max(ymax, yl[1])
    for c in range(2):
        ax = axes.get((r, c))
        if ax is None:
            continue
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, color='k', linewidth=0.4, linestyle='--', alpha=0.3)

add_panel_label(axes[(0, 0)], "a", x=-0.1, y=1.05)

finalize_figure(fig, "figure_S_jl_groups")
