"""Figure 2: Electrode group overview.

Layout:
  Left  (both rows) : Venn diagram — AUD (green) / PROD (blue) / SM overlap
                      (red) — vertical orientation, with a left-hemisphere
                      brain-render inset clipped to each region.
  Right top  (1×3)  : Frequency-averaged z-scored HG power for aud_ls, go_ls,
                      resp (AUD / SM / PROD coloured lines).
  Right bot  (3×3)  : Matching example spectrograms (one electrode per group),
                      sharing the x-axis with the row above.

Data source: stats_freq_hilbert combined LabeledArrays (see slice_freq.py).
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne.time_frequency
import numpy as np
from matplotlib.patches import Circle

from pyvistaqt import BackgroundPlotter

from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray
from ieeg.calc.scaling import rescale
from ieeg.io import get_data
from ieeg.viz.ensemble import plot_dist
from ieeg.viz.mri import plot_on_average
from ieeg.viz.parula import parula_map

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)

folder = "stats_freq_hilbert"
derivatives = os.path.join(layout.root, "derivatives", folder, "combined")

# ---------------------------------------------------------------------------
# Load significance mask  →  groups
# ---------------------------------------------------------------------------
sigs = LabeledArray.fromfile(os.path.join(derivatives, "mask"))
AUD, SM, PROD, sig_chans, delay = group_elecs(
    sigs, sigs.labels[1], sigs.labels[0]
)


# ---------------------------------------------------------------------------
# Load z-scored power
# Full array labels: cond[0], freq[1], channel[2], ...
# After indexing one condition: (freq, channel[, trial], time)
# ---------------------------------------------------------------------------
zscores    = LabeledArray.fromfile(os.path.join(derivatives, "zscore"), mmap_mode="r")
all_chans  = zscores.labels[2]
avail_conds = list(zscores.labels[0])


def freq_avg(cond: str) -> np.ndarray:
    """Return (channel, time) averaged over freq and (if present) trial."""
    raw = zscores[cond].__array__()
    if raw.ndim == 3:
        return np.nanmean(raw, axis=0)        # (freq, ch, t) → (ch, t)
    elif raw.ndim == 4:
        return np.nanmean(raw, axis=(0, 2))   # (freq, ch, trial, t) → (ch, t)
    else:
        keep = {1, raw.ndim - 1}
        return np.nanmean(raw, axis=tuple(i for i in range(raw.ndim) if i not in keep))


def _chans_for_plot(idx_set: set) -> list[str]:
    return [f"D{int(ch.split('-')[0][1:])}-{ch.split('-')[1]}"
            for ch in (all_chans[i] for i in sorted(idx_set))]


def _brain_screenshot(idx_set: set, color: str) -> np.ndarray:
    """Render group electrodes on fsaverage brain, return cropped RGBA array."""
    plotter = BackgroundPlotter(shape=(1, 1))
    brain = plot_on_average(layout.get_subjects(),
                            picks=_chans_for_plot(idx_set),
                            color=color, hemi='lh', show=False)
    for actor in brain.plotter.actors.values():
        plotter.add_actor(actor, reset_camera=False)
    plotter.camera = brain.plotter.camera
    plotter.camera_position = brain.plotter.camera_position
    plotter.view_yz(True)
    plotter.camera.zoom(1.5)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    brain.close()
    if img.dtype != np.uint8:
        try:
            img = (img * 255).astype(np.uint8)
        except Exception:
            img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    rgb = img[..., :3]
    nw = (rgb != 255).any(-1)
    alpha = np.where(nw, 255, 0).astype(np.uint8)
    if img.shape[-1] == 4:
        orig_a = img[..., 3].astype(np.uint8)
        new_a = np.maximum(orig_a, alpha)
        rgba = np.dstack((rgb, new_a))
    else:
        rgba = np.dstack((rgb, alpha))
    rows = np.where(nw.any(1))[0]
    cols = np.where(nw.any(0))[0]
    if rows.size and cols.size:
        return rgba[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return rgba


# ---------------------------------------------------------------------------
# Conditions used in both the line-plot and spectrogram panels
# ---------------------------------------------------------------------------
COND_LIST = [
    ("aud_ls", (-0.5, 1.5), "Stimulus onset"),
    ("go_ls",  (-0.5, 1.5), "Go cue"),
    ("resp",   (-1.0, 1.0), "Response"),
]

GROUPS = [
    ("AUD",  AUD,  "green"),
    ("SM",   SM,   "red"),
    ("PROD", PROD, "blue"),
]

SPEC_TYPE = "multitaper_smooth"
VLIM      = (0.7, 1.4)

# ---------------------------------------------------------------------------
# Representative electrodes for spectrogram rows
# Set each to "D{subj_num}-{channel_name}" or None to auto-select the first
# available electrode in that group (sorted by channel index).
# Examples: "D0005-LTG15", "D0016-MST1", "D0021-LST2"
# ---------------------------------------------------------------------------
SPEC_ELEC = {
    "AUD":  "D0064-lai6",   # e.g. "D0005-LTG15"
    "SM":   "D0028-lpio7",   # e.g. "D0016-MST1"
    "PROD": "D0022-lpif4",   # e.g. "D0021-LST2"
}

# ---------------------------------------------------------------------------
# Figure skeleton
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 14))
gs_outer = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[1, 2],
    wspace=0.28,
)

# ============================================================
# LEFT – Manual vertical Venn diagram spanning the full left column
# ============================================================
ax_venn = fig.add_subplot(gs_outer[0, 0])
ax_venn.set_aspect('equal')
ax_venn.axis('off')

aud_only  = len(AUD  - SM - PROD)
prod_only = len(PROD - SM - AUD)
sm_size   = len(SM)

# Radii proportional to sqrt(count) for approximately area-proportional circles
aud_total  = aud_only + sm_size
prod_total = prod_only + sm_size
max_total  = max(aud_total, prod_total, 1)
R_BASE     = 0.38
r_aud  = R_BASE * np.sqrt(aud_total  / max_total)
r_prod = R_BASE * np.sqrt(prod_total / max_total)

# Center distance: heuristic so geometric overlap is proportional to sm_size
overlap_frac = sm_size / max(aud_total + prod_total - sm_size, 1)
d_centers = (r_aud + r_prod) * max(0.35, 1.0 - 1.4 * overlap_frac)
d_centers = np.clip(d_centers,
                    abs(r_aud - r_prod) + 0.02,
                    r_aud + r_prod - 0.02)

# Vertical layout: AUD on top, PROD on bottom
cy_aud  =  d_centers / 2
cy_prod = -d_centers / 2
cx = 0.0

# Tight axes limits
margin = 0.10
ax_venn.set_xlim(-(max(r_aud, r_prod) + margin), max(r_aud, r_prod) + margin)
ax_venn.set_ylim(cy_prod - r_prod - margin * 1.5,
                 cy_aud  + r_aud  + margin * 1.5)

# Draw AUD circle (green, top)
circ_aud = Circle((cx, cy_aud), r_aud,
                  facecolor='green', alpha=0.10,
                  edgecolor='green', linewidth=2, zorder=1)
ax_venn.add_patch(circ_aud)

# Draw PROD circle (blue, bottom)
circ_prod = Circle((cx, cy_prod), r_prod,
                   facecolor='blue', alpha=0.10,
                   edgecolor='blue', linewidth=2, zorder=1)
ax_venn.add_patch(circ_prod)

# SM intersection (red): draw red AUD-circle clipped to PROD circle extent
sm_patch = Circle((cx, cy_aud), r_aud,
                  facecolor='red', alpha=0.20,
                  edgecolor='none', zorder=2)
ax_venn.add_patch(sm_patch)
sm_patch.set_clip_path(circ_prod)

ax_venn.set_title("Electrode Groups", fontsize=11)

# Force draw so axes limits are settled before computing inset bounds
fig.canvas.draw()

xlim = ax_venn.get_xlim()
ylim = ax_venn.get_ylim()
xr   = xlim[1] - xlim[0]
yr   = ylim[1] - ylim[0]


def _bounds_v(cx_d, cy_d, hw, hh):
    """[x0, y0, w, h] in axes-fraction coordinates for ax_venn."""
    return [(cx_d - hw  - xlim[0]) / xr,
            (cy_d - hh  - ylim[0]) / yr,
            2 * hw / xr,
            2 * hh / yr]


# Geometric midpoints for each region
y_intersect_top    = cy_prod + r_prod   # top of PROD circle (top of intersection)
y_intersect_bottom = cy_aud  - r_aud   # bottom of AUD circle (bottom of intersection)
y_aud_inset  = (y_intersect_top + cy_aud  + r_aud)  / 2   # AUD exclusive centre
y_sm_inset   = (y_intersect_top + y_intersect_bottom) / 2  # intersection centre
y_prod_inset = (cy_prod - r_prod + y_intersect_bottom) / 2 # PROD exclusive centre

inset_hw = min(r_aud, r_prod) * 0.70  # common half-width for all three insets

inset_specs = [
    (cx, y_aud_inset,  inset_hw, AUD,  "green"),
    (cx, y_sm_inset,   inset_hw * 0.80, SM, "red"),
    (cx, y_prod_inset, inset_hw, PROD, "blue"),
]

for icx, icy, hw, idx_set, color in inset_specs:
    shot = _brain_screenshot(idx_set, color)
    h_img, w_img = shot.shape[:2]
    hh = hw * (h_img / w_img)
    ax_ins = ax_venn.inset_axes(_bounds_v(icx, icy, hw, hh), zorder=3)
    ax_ins.set_facecolor('none')
    ax_ins.imshow(shot)
    ax_ins.axis("off")

# Count labels — placed at region centres, above brain insets
lbl_kw = dict(ha='center', va='center', zorder=6,
              bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
ax_venn.text(cx, y_aud_inset,  f"AUD\nn={aud_only}",
             fontsize=9, color='darkgreen', **lbl_kw)
ax_venn.text(cx, y_sm_inset,   f"SM\nn={sm_size}",
             fontsize=9, color='darkred',   **lbl_kw)
ax_venn.text(cx, y_prod_inset, f"PROD\nn={prod_only}",
             fontsize=9, color='darkblue',  **lbl_kw)

# ============================================================
# RIGHT – 4 rows × 3 cols
#   Row 0       : frequency-averaged line plots (1 per condition)
#   Rows 1–3    : spectrograms — one row per group (AUD/SM/PROD),
#                 one col per condition — 3×3 total, single shared colorbar.
# Shared x-axis per column; x-labels only on the bottom row.
# ============================================================
gs_right = gridspec.GridSpecFromSubplotSpec(
    4, 3,
    subplot_spec=gs_outer[0, 1],
    height_ratios=[2, 1, 1, 1],
    hspace=0.06,
    wspace=0.18,
)

axes_ts   = []
axes_spec = [[None] * 3 for _ in range(3)]

# ---- Row 0: frequency-averaged time series ----
ylims_ts = None

for j, (cond, timing, title) in enumerate(COND_LIST):
    ax = fig.add_subplot(gs_right[0, j])
    axes_ts.append(ax)

    if cond in avail_conds:
        arr = freq_avg(cond)
        for grp_name, grp_idx, color in GROUPS:
            idx_list = list(grp_idx)
            if idx_list:
                plot_dist(arr[idx_list], times=timing,
                          label=grp_name, color=color, ax=ax)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlim(timing)
    else:
        ax.text(0.5, 0.5, f"'{cond}' not in data",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)

    ax.set_title(title, fontsize=10)
    plt.setp(ax.get_xticklabels(), visible=False)

    if j == 0:
        ax.set_ylabel("Z-Score HG Power", fontsize=9)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.6)
        ylims_ts = ax.get_ylim()
    else:
        ax.set_yticklabels([])

if ylims_ts:
    for ax in axes_ts:
        ax.set_ylim(ylims_ts)

# ---- Rows 1–3: spectrograms — 3 groups × 3 conditions ----
last_im = None

for i, (grp_name, grp_idx, color) in enumerate(GROUPS):
    if not grp_idx:
        for j in range(3):
            ax = fig.add_subplot(gs_right[i + 1, j])
            ax.text(0.5, 0.5, "no electrodes", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            axes_spec[i][j] = ax
        continue

    # Electrode selection: use SPEC_ELEC override or first available in group
    override = SPEC_ELEC.get(grp_name)
    if override:
        ch_full = override
    else:
        ch_i    = sorted(grp_idx)[0]
        ch_full = all_chans[ch_i]
    subj, ch_name = ch_full.split("-", 1)

    for j, (cond, timing, _title) in enumerate(COND_LIST):
        ax = fig.add_subplot(gs_right[i + 1, j], sharex=axes_ts[j])
        axes_spec[i][j] = ax

        if i == len(GROUPS) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        if j == 0:
            ax.set_ylabel(f"{grp_name}\n{ch_name}\nFreq (Hz)",
                          color=color, fontsize=8)
        else:
            ax.set_yticklabels([])

        tfr_path   = os.path.join(layout.root, "derivatives", "spec",
                                  SPEC_TYPE, subj, f"{cond}-tfr.h5")
        start_path = os.path.join(layout.root, "derivatives", "spec",
                                  SPEC_TYPE, subj, "start-tfr.h5")

        try:
            spec      = mne.time_frequency.read_tfrs(tfr_path)
            start_s   = mne.time_frequency.read_tfrs(start_path)
            spec_avg  = spec.average(lambda x: np.nanmean(x, axis=0))
            start_avg = start_s.average(lambda x: np.nanmean(x, axis=0))
            base      = start_avg.copy().crop(tmin=-0.5, tmax=0)
            rescale(spec_avg, base, mode="ratio", copy=False)

            if ch_name not in spec_avg.ch_names:
                raise ValueError(f"{ch_name} not found in {subj}")

            ch_idx  = spec_avg.ch_names.index(ch_name)
            spec_2d = spec_avg.data[ch_idx]

            im = ax.imshow(
                spec_2d,
                aspect="auto", origin="lower",
                extent=[spec_avg.times[0], spec_avg.times[-1],
                        spec_avg.freqs[0], spec_avg.freqs[-1]],
                cmap=parula_map, vmin=VLIM[0], vmax=VLIM[1],
            )
            last_im = im
            ax.set_xlim(timing)

        except (OSError, ValueError) as exc:
            ax.text(0.5, 0.5, f"{ch_full}\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, wrap=True)
            ax.axis("off")

# Single colorbar spanning all 3 spectrogram rows (right edge of grid)
if last_im is not None:
    right_col = [axes_spec[i][2] for i in range(3)
                 if axes_spec[i][2] is not None]
    if right_col:
        fig.colorbar(last_im, ax=right_col, label="Power ratio", pad=0.02)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.suptitle("Figure 2", fontsize=14, y=0.995)
fig.savefig(os.path.join(out_dir, "figure_2.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_2.png"), bbox_inches="tight", dpi=150)
plt.show()
