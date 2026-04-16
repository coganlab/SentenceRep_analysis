"""Figure 4: SliceTCA Components.

Layout (2 x 3):
  Panel a (row 0, cols 0-1): plot_dist of 4 component time courses for
      aud_ls (left) and go_ls (right), with boxplots of peak latencies.
  Panel b (rows 0-1, col 2): electrode_gradient brain renders (4 x 1)
      showing electrode membership per component.
  Panel c (row 1, cols 0-1): Per-component channel heatmaps — conditions
      arranged horizontally (aud_ls left, go_ls right) to align with
      panel a timings.
"""
import os
import pickle
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import slicetca

from matplotlib.colors import LinearSegmentedColormap, to_rgb

from analysis.figures.config import (
    cm, GS_KWARGS, setup_figure, LABEL_SIZE, TICK_SIZE, DPI,
    COMP_NAMES, COMP_COLORS_LIST, LAYOUT, SM_PKL, SM_MODEL,
)
from analysis.load import load_spec, split_and_stack
from ieeg.viz.ensemble import plot_dist
from ieeg.viz.mri import plot_on_average, _create_color_alpha_matrix
import pyvista as pv

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------
layout = LAYOUT

group = "SM"
folder = "stats_freq_hilbert"
# Condition order must match slice_freq.py: alternating aud/go pairs
conds_ordered = ['aud_ls', 'go_ls', 'aud_lm', 'go_lm', 'aud_jl', 'go_jl']

# Channel labels (pickled from load_spec)
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

# Extract components
W, H = model.get_components(numpy=True)[0]
# W: (n_components, n_channels)   H: (n_components, ..., time)

# Load neural data for channel heatmaps — same condition order as slice_freq.py
neural_data_tensor, mask, _, _ = load_spec(
    group, conds_ordered, layout, folder=folder,
    min_nan=1, n_jobs=-2,
)
# neural_data_tensor: (channels, freqs, trials, time)  time = 6 conds × N pts
# Replicate slice_freq.py processing: split into 3 aud/go pairs, filter transitions
all_con = split_and_stack(neural_data_tensor, -1, 1, 3)
# all_con: (channels, 3_pairs, freqs, trials, N_per_pair)
# pairs: 0=ls, 1=lm, 2=jl; within each pair: aud then go
all_con = torch.cat([all_con[..., :175], all_con[..., 200:375]], dim=-1)
# all_con: (channels, 3_pairs, freqs, trials, 350)  [aud 0:175 | go 175:350]
# Select just the ls pair (index 0)
ls_data = all_con[:, 0]  # (channels, freqs, trials, 350)
# Freq-and-trial averaged: (channels, 350)
data_avg = ls_data.nanmean((1, 2)).detach().cpu().numpy()
# Trial-averaged, keep freq: (channels, freqs, 350)
data_spec = ls_data.nanmean(2).detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
colors = COMP_COLORS_LIST
names = COMP_NAMES

# Per-component colormaps: white (0) → component color (max)
comp_cmaps = []
for c in colors:
    rgb = to_rgb(c)
    comp_cmaps.append(LinearSegmentedColormap.from_list(
        f"white_{c}", [(1, 1, 1), rgb], N=256
    ))
timings = {"aud_ls": slice(0, 175), "go_ls": slice(175, 350)}
conds_plot = {"aud_ls": (-0.5, 1.25), "go_ls": (-0.5, 1.25)}
chans = [
    "-".join([f"D{int(ch.split('-')[0][1:])}", ch.split("-")[1]])
    for ch in labels[0]
]

# ---------------------------------------------------------------------------
# Figure skeleton
# ---------------------------------------------------------------------------
# Outer split: left panels (a + c) vs right panel (b)
fig = setup_figure(figsize=(24 * cm, 18 * cm))
gs_outer = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[2, 0.7],
    wspace=0.2,
)

# Left side: single gridspec shared by panels a (row 0) and c (rows 1+)
# so that their two columns are guaranteed to align exactly.
# 3 columns: col 0 = aud_ls, col 1 = colorbars (slim), col 2 = go_ls
n_rows_c = n_components * 2
gs_left = gridspec.GridSpecFromSubplotSpec(
    1 + n_rows_c, 3, subplot_spec=gs_outer[0, 0],
    height_ratios=[1] + [0.175] * n_rows_c,
    width_ratios=[1, 0.03, 1],
    hspace=0.25, wspace=0.08,
)

# =================== Panel a (row 0): time courses =========================
COL_IDX = [0, 2]  # data columns in gs_left (skip col 1 = colorbar)

ylims = [0.0, 0.0]
axes_a = []
for j, (cond, times) in enumerate(conds_plot.items()):
    ax = fig.add_subplot(gs_left[0, COL_IDX[j]])
    axes_a.append(ax)
    n_time = len(range(*timings[cond].indices(H.shape[-1])))
    for i in range(n_components):
        component_data = H[i, ..., timings[cond]].reshape(-1, n_time)
        plot_dist(component_data, ax=ax, color=colors[i], mode="sem",
                  times=times, label=names[i])
    ax.set_xlim(times)
    plt.setp(ax.get_xticklabels(), visible=False)
    if j == 0:
        ax.set_ylabel("HG Power (z)", fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper left")
    else:
        ax.set_yticklabels([])
    yl = ax.get_ylim()
    ylims[0] = min(ylims[0], yl[0])
    ylims[1] = max(ylims[1], yl[1])

# Unify y-limits and add peak-time boxplots
for j, (cond, times) in enumerate(conds_plot.items()):
    ax = axes_a[j]
    ax.set_ylim(ylims)
    n_time = len(range(*timings[cond].indices(H.shape[-1])))
    positions = np.linspace(
        (ylims[0] + ylims[1]) * 4 / 5, ylims[1], n_components
    )
    width = positions[1] - positions[0]
    positions -= width / 2
    for i in range(n_components):
        comp_data = H[i, ..., timings[cond]].reshape(-1, n_time)
        ttimes = np.linspace(times[0], times[1], comp_data.shape[-1])
        peak_times = ttimes[comp_data.argmax(axis=-1)]
        ax.boxplot(
            peak_times, vert=False, manage_ticks=False,
            positions=[positions[i]], widths=width / 2,
            patch_artist=True, boxprops=dict(facecolor=colors[i]),
            medianprops=dict(color="k", alpha=0.5), showfliers=False,
        )

# ============= Panel b (right column): brain renders (4×1) ================
max_size = 2.0
scale_W = W.copy()
scale_W[scale_W > max_size] = max_size
comp_sizes = scale_W / 2
comp_colors = _create_color_alpha_matrix(
    colors[:n_components], scale_W / scale_W.max()
)

gs_b = gridspec.GridSpecFromSubplotSpec(
    n_components, 1, subplot_spec=gs_outer[0, 1],
    hspace=0.05,
)

subjects = layout.get_subjects()
ax_brain_first = None
for i in range(n_components):
    ax_b = fig.add_subplot(gs_b[i, 0])
    if ax_brain_first is None:
        ax_brain_first = ax_b
    ax_b.axis("off")
    brain = plot_on_average(subjects, picks=list(chans),
                            size=comp_sizes[i], hemi='both',
                            color=comp_colors[i], show=False,
                            transparency=0.15)
    plotter = pv.Plotter(off_screen=True, window_size=(3200, 2400))
    for actor in brain.plotter.actors.values():
        plotter.add_actor(actor, reset_camera=False)
    plotter.camera = brain.plotter.camera
    plotter.camera_position = brain.plotter.camera_position
    plotter.view_yz(True)
    plotter.render()
    img = plotter.screenshot(return_img=True)
    plotter.close()
    brain.close()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255) if img.max() <= 255
               else np.clip(img * 255, 0, 255)).astype(np.uint8)
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    rgb = img[..., :3]
    fg = (rgb != 255).any(-1)
    rows = np.where(fg.any(1))[0]
    cols = np.where(fg.any(0))[0]
    if rows.size and cols.size:
        img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    ax_b.imshow(img)
    ax_b.set_ylabel(names[i], fontsize=LABEL_SIZE, rotation=0,
                    labelpad=20, va="center")

# ========= Panel c (rows 1+): channel heatmaps ============================
# Uses gs_left rows 1..n so columns align exactly with panel a above.
freqs = np.logspace(np.log10(50), np.log10(300), data_spec.shape[1])
SPEC_VLIM = (0, 0.35)
n_conds = len(timings)

ax_c_first = None
cond_list = list(timings.items())
for i in range(n_components):
    membership = (W[i] / W.sum(0)) > 0.4
    w_mem = W[i, membership]
    spec_im = None
    heat_im = None
    heat_scale = None
    for j_cond, (cond, tslice) in enumerate(cond_list):
        time_range = range(tslice.start, tslice.stop)
        trimmed = data_avg[membership][:, time_range]
        sorted_trimmed = trimmed[np.argsort(W[i, membership])][::-1]

        # Spectrogram — row offset 1 to skip panel a row
        ax_spec = fig.add_subplot(gs_left[1 + i * 2, COL_IDX[j_cond]],
                                  sharex=axes_a[j_cond])
        if ax_c_first is None:
            ax_c_first = ax_spec
        spec_data = data_spec[membership][:, :, time_range]
        spec_weighted = np.average(spec_data, axis=0, weights=w_mem)
        spec_im = ax_spec.imshow(
            spec_weighted, aspect="auto", origin="lower",
            cmap=comp_cmaps[i], vmin=SPEC_VLIM[0], vmax=SPEC_VLIM[1],
            extent=[conds_plot[cond][0], conds_plot[cond][1],
                    freqs[0], freqs[-1]],
        )
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        if j_cond > 0:
            ax_spec.set_yticks([])
            ax_spec.set_yticklabels([])
        elif i == 0:
            ax_spec.set_ylabel("Freq (Hz)", fontsize=LABEL_SIZE)

        # Heatmap
        ax_heat = fig.add_subplot(gs_left[1 + i * 2 + 1, COL_IDX[j_cond]],
                                  sharex=axes_a[j_cond])
        heat_scale = np.mean(sorted_trimmed) + 1.5 * np.std(sorted_trimmed)
        heat_im = ax_heat.imshow(
            sorted_trimmed, aspect="auto", cmap="inferno",
            vmin=0, vmax=heat_scale,
            extent=[conds_plot[cond][0], conds_plot[cond][1],
                    0, len(sorted_trimmed)],
        )
        if j_cond > 0:
            ax_heat.set_yticks([])
            ax_heat.set_yticklabels([])
        elif i == 0:
            ax_heat.set_ylabel("Channels", fontsize=LABEL_SIZE)
        # x-label only on the bottom row
        if i == n_components - 1:
            if cond.startswith("go"):
                event_label = "Go Cue"
            else:
                event_label = "Stimulus"
            ax_heat.set_xlabel(f"Time from {event_label} (s)", fontsize=LABEL_SIZE)
        else:
            plt.setp(ax_heat.get_xticklabels(), visible=False)

    # Colorbars: shift rightward from gridspec col 1 so tick labels don't overlap
    # Spectrogram colorbar
    ax_cb_ref = fig.add_subplot(gs_left[1 + i * 2, 1])
    ax_cb_ref.set_visible(False)
    bbox = ax_cb_ref.get_position()
    cax_spec = fig.add_axes([bbox.x0 + bbox.width, bbox.y0,
                             bbox.width, bbox.height])
    fig.colorbar(spec_im, cax=cax_spec)
    cax_spec.yaxis.set_ticks_position("left")
    # Heatmap colorbar
    ax_cb_ref2 = fig.add_subplot(gs_left[1 + i * 2 + 1, 1])
    ax_cb_ref2.set_visible(False)
    bbox2 = ax_cb_ref2.get_position()
    cax_heat = fig.add_axes([bbox2.x0 + bbox2.width, bbox2.y0,
                             bbox2.width, bbox2.height])
    fig.colorbar(heat_im, cax=cax_heat)
    cax_heat.yaxis.set_ticks_position("left")

# ---------------------------------------------------------------------------
# Subfigure labels
# ---------------------------------------------------------------------------
axes_a[0].text(-0.15, 1.1, "a", transform=axes_a[0].transAxes,
               fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")
ax_brain_first.text(-0.1, 1.1, "b", transform=ax_brain_first.transAxes,
                    fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")
ax_c_first.text(-0.2, 1.1, "c", transform=ax_c_first.transAxes,
                fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_4.svg"), bbox_inches="tight", dpi=DPI)
fig.savefig(os.path.join(out_dir, "figure_4.png"), bbox_inches="tight", dpi=DPI)
plt.show()
