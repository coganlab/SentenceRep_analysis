"""Figure 4: SliceTCA Components.

Layout:
  Top row        : plot_dist of 4 component time courses for aud_ls (left)
                   and go_ls (right), with boxplots of peak latencies.
  Bottom-left    : Per-component channel heatmaps — for each condition
                   (aud_ls / go_ls), a parula spectrogram (component-weighted
                   mean) and sorted-channel imshow below it.
  Bottom-right   : electrode_gradient brain renders showing electrode
                   membership per component.
"""
import os
import pickle
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cuda.preferred_linalg_library("cusolver")
import slicetca

from matplotlib.colors import LinearSegmentedColormap, to_rgb

from analysis.figures.config import cm, GS_KWARGS, setup_figure, LABEL_SIZE, TICK_SIZE
from analysis.load import load_spec, split_and_stack
from ieeg.io import get_data
from ieeg.viz.ensemble import plot_dist
from ieeg.viz.mri import plot_on_average, _create_color_alpha_matrix
import pyvista as pv

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)

group = "SM"
folder = "stats_freq_hilbert"
# Condition order must match slice_freq.py: alternating aud/go pairs
conds_ordered = ['aud_ls', 'go_ls', 'aud_lm', 'go_lm', 'aud_jl', 'go_jl']

# Channel labels (pickled from load_spec)
pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "SM_chns.pkl")
with open(pkl_path, "rb") as f:
    labels = pickle.load(f)

# Reconstruct model from saved state dict
state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "model_SM2_freq.pt")
state = torch.load(state_path, map_location="cpu")
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
colors = ["orange", "k", "c", "y"]
names = ["Auditory", "WM", "Motor", "Visual"]

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
fig = setup_figure()
gs_outer = gridspec.GridSpec(
    2, 1, figure=fig, height_ratios=[1, 1.2],
    hspace=0.3, wspace=GS_KWARGS['wspace'],
)

# ========================== TOP: component time courses ====================
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[0],
    hspace=0.3, wspace=GS_KWARGS['wspace'],
)

ylims = [0.0, 0.0]
axes_top = []
for j, (cond, times) in enumerate(conds_plot.items()):
    ax = fig.add_subplot(gs_top[0, j])
    axes_top.append(ax)
    n_time = len(range(*timings[cond].indices(H.shape[-1])))
    for i in range(n_components):
        component_data = H[i, ..., timings[cond]].reshape(-1, n_time)
        plot_dist(component_data, ax=ax, color=colors[i], mode="sem",
                  times=times, label=names[i])
    if cond.startswith("go"):
        event = "Go Cue"
    else:
        event = "Stimulus"
    ax.set_xlabel(f"Time(s) from {event}", fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    if j == 0:
        ax.set_ylabel("Z-Score HG power", fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper left")
    yl = ax.get_ylim()
    ylims[0] = min(ylims[0], yl[0])
    ylims[1] = max(ylims[1], yl[1])

# Unify y-limits and add peak-time boxplots
for j, (cond, times) in enumerate(conds_plot.items()):
    ax = axes_top[j]
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

# ========================= BOTTOM: channels + brains ======================
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[1], width_ratios=[1.2, 1],
    hspace=0.3, wspace=GS_KWARGS['wspace'],
)

# ---------- Bottom-left: channel heatmaps ----------
# 5 rows: spec1, heat1, (spacer), spec2, heat2
# height_ratios: equal for data rows, larger spacer for x-label gap
n_conds = len(timings)
gs_chans = gridspec.GridSpecFromSubplotSpec(
    2 * n_conds + 1, n_components, subplot_spec=gs_bot[0, 0],
    height_ratios=[1, 1, 0.6, 1, 1],
    hspace=0.15, wspace=GS_KWARGS['wspace'],
)

# Frequency axis for spectrograms (31 log-spaced bins from 50–300 Hz)
freqs = np.logspace(np.log10(50), np.log10(300), data_spec.shape[1])
SPEC_VLIM = (0, 0.35)

ax_bot_left_first = None   # will hold the first bottom-left axis for label "b"
# Row mapping: cond 0 → rows 0,1; cond 1 → rows 3,4 (row 2 is spacer)
row_offsets = [0, 3]
heat_axes_by_cond = {cond: [] for cond in timings}  # track heatmap axes per block

for j_cond, (cond, tslice) in enumerate(timings.items()):
    j = row_offsets[j_cond]
    time_range = range(tslice.start, tslice.stop)
    if cond.startswith("go"):
        event_label = "Go Cue"
    else:
        event_label = "Stimulus"
    for i in range(n_components):
        # Select channels belonging to this component
        membership = (W[i] / W.sum(0)) > 0.4
        trimmed = data_avg[membership][:, time_range]
        sorted_trimmed = trimmed[np.argsort(W[i, membership])][::-1]

        # Spectrogram (component-weighted mean across member channels)
        ax_spec = fig.add_subplot(gs_chans[j, i])
        if ax_bot_left_first is None:
            ax_bot_left_first = ax_spec
        # data_spec: (channels, freqs, time) — select member channels & time
        spec_data = data_spec[membership][:, :, time_range]
        # Weight by component weights and average: (freqs, time)
        w_mem = W[i, membership]
        spec_weighted = np.average(spec_data, axis=0, weights=w_mem)
        ax_spec.imshow(
            spec_weighted, aspect="auto", origin="lower",
            cmap=comp_cmaps[i], vmin=SPEC_VLIM[0], vmax=SPEC_VLIM[1],
            extent=[conds_plot[cond][0], conds_plot[cond][1],
                    freqs[0], freqs[-1]],
        )
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        ax_spec.tick_params(axis='both', labelsize=TICK_SIZE)
        if i > 0:
            ax_spec.set_yticks([])
            ax_spec.set_yticklabels([])
        else:
            ax_spec.set_ylabel("Freq (Hz)", fontsize=6)

        # Heatmap
        ax_heat = fig.add_subplot(gs_chans[j + 1, i])
        scale = np.mean(sorted_trimmed) + 1.5 * np.std(sorted_trimmed)
        ax_heat.imshow(
            sorted_trimmed, aspect="auto", cmap="inferno",
            vmin=0, vmax=scale,
            extent=[conds_plot[cond][0], conds_plot[cond][1],
                    0, len(sorted_trimmed)],
        )
        ax_heat.tick_params(axis='both', labelsize=TICK_SIZE)
        heat_axes_by_cond[cond].append(ax_heat)
        if i > 0:
            ax_heat.set_yticks([])
            ax_heat.set_yticklabels([])
        else:
            ax_heat.set_ylabel("Channels", fontsize=6)

# Shared x-axis labels centred across all 4 columns per condition block
fig.canvas.draw()
for cond, axes_list in heat_axes_by_cond.items():
    if cond.startswith("go"):
        event_label = "Go Cue"
    else:
        event_label = "Stimulus"
    # Compute centre x and bottom y from the row of heatmap axes (in fig coords)
    bboxes = [ax.get_position() for ax in axes_list]
    x_center = (bboxes[0].x0 + bboxes[-1].x1) / 2
    y_bottom = min(bb.y0 for bb in bboxes)
    fig.text(x_center, y_bottom - 0.02, f"Time(s) from {event_label}",
             fontsize=6, ha="center", va="top")

# ---------- Bottom-right: brain renders per component (2×2) ----------
# Replicate electrode_gradient logic: per-component plotter, screenshot,
# autocrop, embed into matplotlib axes.
max_size = 2.0
scale_W = W.copy()
scale_W[scale_W > max_size] = max_size
comp_sizes = scale_W / 2                          # size per electrode
comp_colors = _create_color_alpha_matrix(          # faded RGBA per electrode
    colors[:n_components], scale_W / scale_W.max()
)

gs_brains = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_bot[0, 1],
    hspace=0.15, wspace=GS_KWARGS['wspace'],
)

subjects = layout.get_subjects()
ax_brain_first = None   # will hold the first brain axis for label "c"
for i in range(n_components):
    ax_b = fig.add_subplot(gs_brains[i // 2, i % 2])
    if ax_brain_first is None:
        ax_brain_first = ax_b
    ax_b.axis("off")
    # Render this component's electrodes on fsaverage
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
    # Normalise & autocrop
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

# ---------------------------------------------------------------------------
# Remove top and right spines from all axes
# ---------------------------------------------------------------------------
for ax in fig.get_axes():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------------
# Subfigure labels
# ---------------------------------------------------------------------------
axes_top[0].text(-0.15, 1.1, "a", transform=axes_top[0].transAxes,
                 fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")
ax_bot_left_first.text(-0.2, 1.1, "b", transform=ax_bot_left_first.transAxes,
                       fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")
ax_brain_first.text(-0.1, 1.1, "c", transform=ax_brain_first.transAxes,
                    fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_4.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_4.png"), bbox_inches="tight", dpi=150)
plt.show()
