"""Supplementary Figure: Pairwise Component Weight Scatter Plots.

Layout (6 x 3):
  Each row is one of the 6 pairwise component combinations.
  Column 0: Brain render showing pairwise electrode membership.
  Column 1: Log-weight scatter plot with threshold lines.
  Column 2: Pie chart of scatter-plot quadrant membership.
"""
import os
import pickle
from functools import partial
from itertools import combinations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import slicetca
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE, DPI,
    COMP_NAMES, COMP_COLORS_LIST, LAYOUT, SM_PKL, SM_MODEL,
)
from ieeg.viz.mri import (
    electrode_ratio_gradient, ratio_to_color_gradient,
)

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------
layout = LAYOUT

# Channel labels
with open(SM_PKL, "rb") as f:
    labels = pickle.load(f)

# Reconstruct model from saved state dict
state = torch.load(SM_MODEL, map_location="cpu")
shape = state["vectors.0.0"].shape[1:] + state["vectors.0.1"].shape[1:]
n_components = state["vectors.0.0"].shape[0]
model = slicetca.core.SliceTCA(
    dimensions=shape,
    ranks=(n_components, 0, 0, 0),
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
colors = COMP_COLORS_LIST
names = COMP_NAMES
THRESH = -6.5  # log-weight threshold

chans = [
    "-".join([f"D{int(ch.split('-')[0][1:])}", ch.split("-")[1]])
    for ch in labels[0]
]

# Log-transform weights
W_log = np.log(W + 1e-10)

# All 6 pairwise combinations
pairs = list(combinations(range(n_components), 2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def autocrop_nonwhite(img, *, white_thresh=250):
    """Crop to bounding box containing any non-background pixels."""
    rgb = img[..., :3]
    fg = (rgb < white_thresh).any(-1)
    rows = np.where(fg.any(1))[0]
    cols = np.where(fg.any(0))[0]
    if rows.size and cols.size:
        return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return img


def color_contrast(rgba, weights):
    """Grey-out colours based on weight (0=full color, 1=grey)."""
    greyed = []
    grey = 0.5
    for color, w in zip(rgba, weights):
        gc = tuple(grey * w + c * (1 - w) for c in color[:3]) + (color[3],)
        greyed.append(gc)
    return greyed


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(18 * cm, 36 * cm))
gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3,
                       width_ratios=[1, 1, 0.8])

for row, (w1, w2) in enumerate(pairs):
    # --- Column 0: Brain render ---
    ax_brain = fig.add_subplot(gs[row, 0])
    ax_brain.axis("off")

    cmap_pair = LinearSegmentedColormap.from_list(
        "two_color_cmap", [colors[w2], colors[w1]], N=200)
    weights_2 = np.stack([W[w1], W[w2]], axis=0)
    plotter = electrode_ratio_gradient(
        layout.get_subjects(), weights_2, chans, cmap_pair, 2, 16, (1, 1))
    img = plotter.screenshot(return_img=True)
    plotter.close()
    img = autocrop_nonwhite(img)
    ax_brain.imshow(img)
    if row == 0:
        ax_brain.set_title("Brain", fontsize=LABEL_SIZE)

    # --- Column 1: Scatter plot ---
    ax_scat = fig.add_subplot(gs[row, 1])

    cmap_scat = LinearSegmentedColormap.from_list(
        "two_color_cmap", [colors[w2], colors[w1]], N=200)
    c = ratio_to_color_gradient(W_log[w2], W_log[w1], colormap=cmap_scat)
    Wt = np.max([W_log[w1], W_log[w2]], axis=0)
    Wt -= Wt.min()
    Wt /= Wt.max()
    wgts = 1 - Wt
    c_greyed = color_contrast(list(c), list(wgts))
    ax_scat.scatter(W_log[w1], W_log[w2], c=c_greyed, s=4)
    ax_scat.axvline(THRESH, color="k", linestyle="--", linewidth=0.5)
    ax_scat.axhline(THRESH, color="k", linestyle="--", linewidth=0.5)
    combo_title = f"{names[w1]} + {names[w2]}"
    ax_scat.set_title(combo_title, fontsize=LABEL_SIZE)

    # --- Column 2: Pie chart of scatter quadrants ---
    ax_pie = fig.add_subplot(gs[row, 2])

    hi_w1 = W_log[w1] > THRESH
    hi_w2 = W_log[w2] > THRESH
    counts = [
        np.sum(hi_w1 & hi_w2),       # both
        np.sum(hi_w1 & ~hi_w2),      # w1 only
        np.sum(~hi_w1 & hi_w2),      # w2 only
        np.sum(~hi_w1 & ~hi_w2),     # neither
    ]
    pie_labels = [combo_title, names[w1], names[w2], "Neither"]
    pie_colors = [cmap_pair(0.5), colors[w1], colors[w2], "lightgrey"]

    # Only plot slices with nonzero count
    nonzero = [i for i, c in enumerate(counts) if c > 0]
    ax_pie.pie(
        [counts[i] for i in nonzero],
        labels=[pie_labels[i] for i in nonzero],
        colors=[pie_colors[i] for i in nonzero],
        autopct=lambda pct: f"{int(np.round(pct / 100 * sum(counts)))}",
        textprops={"fontsize": TICK_SIZE},
        startangle=90,
        counterclock=True,
    )
    if row == 0:
        ax_pie.set_title("Quadrant\nMembership", fontsize=LABEL_SIZE)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_S_weight_scatter.svg"),
            bbox_inches="tight", dpi=DPI)
fig.savefig(os.path.join(out_dir, "figure_S_weight_scatter.png"),
            bbox_inches="tight", dpi=DPI)
plt.show()
