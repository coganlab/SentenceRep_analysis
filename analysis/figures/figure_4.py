"""Figure 4: SliceTCA Components.

Layout:
  Top row        : plot_dist of 4 component time courses for aud_ls (left)
                   and go_ls (right), with boxplots of peak latencies.
  Bottom-left    : Per-component channel heatmaps — for each condition
                   (aud_ls / go_ls), a plot_dist line and sorted-channel
                   imshow below it.
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
import slicetca

from analysis.figures.config import cm, GS_KWARGS, setup_figure, LABEL_SIZE, TICK_SIZE
from analysis.load import load_spec, split_and_stack
from ieeg.io import get_data
from ieeg.viz.ensemble import plot_dist
from ieeg.viz.mri import plot_on_average, _create_color_alpha_matrix
from pyvistaqt import BackgroundPlotter

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)

group = "SM"
folder = "stats_freq_hilbert"
conds_all = {
    "aud_ls": (-0.5, 1.5), "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
    "go_ls": (-0.5, 1.5),  "go_lm": (-0.5, 1.5),  "go_jl": (-0.5, 1.5),
}

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

# Load neural data for channel heatmaps
neural_data_tensor, mask, _, _ = load_spec(
    group, list(conds_all.keys()), layout, folder=folder,
    min_nan=1, n_jobs=-2,
)
# neural_data_tensor: (channels, freqs, trials, time)
# Freq-and-trial averaged: (channels, time)  — time = 6 conds × 200 pts
data_avg = neural_data_tensor[..., :375].nanmean((1, 2)).detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
colors = ["orange", "k", "c", "y"]
names = ["Auditory", "WM", "Motor", "Visual"]
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
    2, 1, figure=fig, height_ratios=[1, 1.2], **GS_KWARGS,
)

# ========================== TOP: component time courses ====================
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[0], **GS_KWARGS,
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
    1, 2, subplot_spec=gs_outer[1], width_ratios=[1.2, 1], **GS_KWARGS,
)

# ---------- Bottom-left: channel heatmaps ----------
n_conds = len(timings)
gs_chans = gridspec.GridSpecFromSubplotSpec(
    2 * n_conds, n_components, subplot_spec=gs_bot[0, 0],
    **GS_KWARGS,
)

for j_cond, (cond, tslice) in enumerate(timings.items()):
    j = j_cond * 2  # row offset (0 or 2)
    time_range = range(tslice.start, tslice.stop)
    ylims_ch = [0.0, 0.0]
    line_axes = []
    for i in range(n_components):
        # Select channels belonging to this component
        membership = (W[i] / W.sum(0)) > 0.4
        trimmed = data_avg[membership][:, time_range]
        sorted_trimmed = trimmed[np.argsort(W[i, membership])][::-1]

        # Line plot
        ax_line = fig.add_subplot(gs_chans[j, i])
        line_axes.append(ax_line)
        plot_dist(trimmed.reshape(-1, len(time_range)), ax=ax_line,
                  color=colors[i], mode="sem", times=conds_plot[cond])
        ax_line.set_xticks([])
        ax_line.set_xticklabels([])
        ylims_ch[0] = min(ylims_ch[0], ax_line.get_ylim()[0])
        ylims_ch[1] = max(ylims_ch[1], ax_line.get_ylim()[1])
        ax_line.tick_params(axis='both', labelsize=TICK_SIZE)
        if i > 0:
            ax_line.set_yticks([])
            ax_line.set_yticklabels([])
        else:
            ax_line.set_ylabel("Z-Score", fontsize=LABEL_SIZE)

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
        if i > 0:
            ax_heat.set_yticks([])
            ax_heat.set_yticklabels([])
        else:
            ax_heat.set_ylabel("Channels", fontsize=LABEL_SIZE)
        if j_cond == 0:
            ax_heat.set_xticklabels([])

    # Unify y-limits for line plots in this condition
    for ax_l in line_axes:
        ax_l.set_ylim(ylims_ch)

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
    2, 2, subplot_spec=gs_bot[0, 1], **GS_KWARGS,
)

subjects = layout.get_subjects()
for i in range(n_components):
    ax_b = fig.add_subplot(gs_brains[i // 2, i % 2])
    ax_b.axis("off")
    # Render this component's electrodes on fsaverage
    plotter = BackgroundPlotter(shape=(1, 1))
    brain = plot_on_average(subjects, picks=list(chans),
                            size=comp_sizes[i], hemi='both',
                            color=comp_colors[i], show=False,
                            transparency=0.15)
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
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_4.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_4.png"), bbox_inches="tight", dpi=150)
plt.show()
