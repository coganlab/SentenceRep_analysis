"""Supplementary Figure: ROI breakdown of the four sliceTCA components.

For each component (Auditory / WM / Motor / Visual) counts how many
electrodes (those with normalised weight ``W[i]/W.sum(0) > 0.4``) fall
into each Brainnetome gyrus-level ROI and draws a per-component bar chart.
Mirrors ``analysis/decomposition/slice_freq.py:840-909``.
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
    LAYOUT, SM_PKL, SM_MODEL,
    COMP_NAMES, COMP_COLORS_LIST,
    finalize_figure, add_panel_label,
)
from ieeg.viz.mri import gen_labels, subject_to_info, Atlas

# ---------------------------------------------------------------------------
# Load the sliceTCA solution
# ---------------------------------------------------------------------------
layout = LAYOUT
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
W, _ = model.get_components(numpy=True)[0]
# Exclusive membership: log(W + 1e-10) > -6.5 AND component is argmax.
# This makes each electrode contribute to at most one component, so the
# per-component ROI counts are disjoint. Mirrors
# analysis/results/utils.py:component_membership_exclusive.
LOGW_THRESHOLD = -6.5
inclusive = np.log(W + 1e-10) > LOGW_THRESHOLD
argmax_i = W.argmax(axis=0)
members = np.zeros_like(inclusive)
for c in range(W.shape[1]):
    i = argmax_i[c]
    if inclusive[i, c]:
        members[i, c] = True

# SM electrode labels in the form 'D{int}-{ch}' for matching to gen_labels keys
chan_strs = [
    "-".join([f"D{int(c.split('-')[0][1:])}", c.split("-")[1]])
    for c in labels[0]
]

# Brainnetome ROIs to count, copied from slice_freq.py
ROIS = [
    'IFG', 'Tha', 'PoG', 'Amyg', 'PhG', 'MVOcC', 'ITG', 'PrG', 'PCL',
    'IPL', 'MFG', 'CG', 'Pcun', 'BG', 'INS', 'FuG', 'LOcC', 'STG',
    'OrG', 'MTG', 'pSTS', 'Hipp', 'SFG', 'SPL',
]
atlas = Atlas()


def _count_rois(member_mask: np.ndarray) -> dict[str, int]:
    target = {chan_strs[i] for i in np.nonzero(member_mask)[0]}
    counts = {r: 0 for r in ROIS}
    for subj in layout.get_subjects():
        subj_old = f"D{int(subj[1:])}"
        try:
            info = subject_to_info(subj_old)
        except Exception:
            continue
        ch_labels = gen_labels(info, subj_old, atlas='.BN_atlas')
        for ch, val in ch_labels.items():
            full = f"{subj_old}-{ch}"
            if full not in target:
                continue
            base = val.split("_")[0]
            try:
                gyrus = atlas[base].gyrus
            except KeyError:
                if base == 'TE1.0/TE1.2':
                    gyrus = 'STG'
                else:
                    continue
            if gyrus in counts:
                counts[gyrus] += 1
    return counts


comp_counts = []
for i in range(n_components):
    membership = members[i]
    counts = _count_rois(membership)
    n = sum(counts.values())
    comp_counts.append((COMP_NAMES[i], COMP_COLORS_LIST[i], counts, n))

# Keep only the top 10 ROIs by total count across all components
TOP_N = 10
totals = {r: sum(c[r] for _, _, c, _ in comp_counts) for r in ROIS}
keep_rois = sorted([r for r in ROIS if totals[r] > 0],
                   key=lambda r: -totals[r])[:TOP_N]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(18 * cm, 14 * cm))
gs = gridspec.GridSpec(n_components, 1, figure=fig, hspace=0.35)

ymax = max(c[r] for _, _, c, _ in comp_counts for r in keep_rois)
for i, (name, color, counts, n) in enumerate(comp_counts):
    ax = fig.add_subplot(gs[i, 0])
    ax.bar(keep_rois, [counts[r] for r in keep_rois], color=color)
    ax.set_ylabel(f"{name}\n(n={n})", fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_ylim(0, ymax * 1.05)
    if i == n_components - 1:
        ax.set_xticks(range(len(keep_rois)))
        ax.set_xticklabels(keep_rois, rotation=45, ha='right',
                           fontsize=TICK_SIZE)
    else:
        ax.set_xticks([])
    add_panel_label(ax, "abcd"[i], x=-0.06, y=1.05)
    print(f"\n{name} (n={n})")
    for r in keep_rois:
        if counts[r]:
            print(f"  {r:<6} {counts[r]:>4}")

ax.set_xlabel("Brainnetome gyrus-level ROI", fontsize=LABEL_SIZE)

finalize_figure(fig, "figure_S_roi_components")
