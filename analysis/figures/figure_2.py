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
import mne
import mne.time_frequency
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import brentq

from analysis.figures.config import (
    cm, LABEL_SIZE, TICK_SIZE, GS_KWARGS, setup_figure, LAYOUT, DPI,
    EVENT_STIMULUS, EVENT_GO, EVENT_RESPONSE, POWER_RATIO_LABEL,
    finalize_figure, add_panel_label,
)
import pyvista as pv

from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray
from ieeg.calc.scaling import rescale
from ieeg.viz.ensemble import plot_dist
from ieeg.viz.mri import force2frame, get_sub_dir, plot_on_average, subject_to_info
from ieeg.viz.parula import parula_map

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
layout = LAYOUT

folder = 'stats_freq_hilbert'

conds_all = {"resp": (-1., 1.), "aud_ls": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5)}

# zscores = load_data(layout, folder, "zscore", conds_all, 3,
#                     "float16", True, 1, "combined2")
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
        return np.nanmean(raw, axis=1)        # (freq, ch, t) → (ch, t)
    elif raw.ndim == 4:
        return np.nanmean(raw, axis=(0, 2))   # (freq, ch, trial, t) → (ch, t)
    else:
        keep = {1, raw.ndim - 1}
        return np.nanmean(raw, axis=tuple(i for i in range(raw.ndim) if i not in keep))


def _chans_for_plot(idx_set: set) -> list[str]:
    return [f"D{int(ch.split('-')[0][1:])}-{ch.split('-')[1]}"
            for ch in (all_chans[i] for i in sorted(idx_set))]


def _brain_screenshot(idx_set: set, color: str,
                      highlight_elecs: dict = None):
    """Render group electrodes on fsaverage brain; return (cropped RGBA, markers).

    Parameters
    ----------
    highlight_elecs : dict, optional
        ``{symbol: 'D0028-LPIO7'}`` — each electrode's 3-D position is
        projected to 2-D screen coords via VTK WorldToDisplay *before*
        screenshotting, so the caller can overlay text markers on the inset.

    Returns
    -------
    rgba : np.ndarray  — cropped RGBA screenshot
    marker_px : list of (grp_name, px, py) in cropped-image pixel coordinates
    """
    brain = plot_on_average(layout.get_subjects(),
                            picks=_chans_for_plot(idx_set),
                            color=color, hemi='lh', show=False)
    plotter = pv.Plotter(off_screen=True, window_size=(3200, 2400))
    for actor in brain.plotter.actors.values():
        plotter.add_actor(actor, reset_camera=False)
    plotter.camera = brain.plotter.camera
    plotter.camera_position = brain.plotter.camera_position
    plotter.view_yz(True)
    plotter.camera.zoom(1.5)
    plotter.render()  # required before WorldToDisplay matrices are valid

    # Project each highlighted electrode to 2-D image pixel coordinates
    marker_px = []
    if highlight_elecs:
        ren = plotter.renderer
        win_h = plotter.window_size[1]
        for elec_key, elec_str in highlight_elecs.items():
            pos_3d = _get_elec_fsaverage_pos(elec_str)
            if pos_3d is None:
                continue
            ren.SetWorldPoint(float(pos_3d[0]), float(pos_3d[1]),
                              float(pos_3d[2]), 1.0)
            ren.WorldToDisplay()
            dp = ren.GetDisplayPoint()
            # VTK display: (0,0) = bottom-left; image array: (0,0) = top-left
            marker_px.append((elec_key, float(dp[0]), float(win_h - dp[1])))

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
        rgba = rgba[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        # Shift marker coords to account for the crop
        marker_px = [(k, px - cols[0], py - rows[0])
                     for k, px, py in marker_px]
    return rgba, marker_px


# ---------------------------------------------------------------------------
# Conditions used in both the line-plot and spectrogram panels
# ---------------------------------------------------------------------------
COND_LIST = [
    ("aud_ls", (-0.5, 1.5), EVENT_STIMULUS),
    ("go_ls",  (-0.5, 1.5), EVENT_GO),
    ("resp",   (-1.0, 1.0), EVENT_RESPONSE),
]

GROUPS = [
    ("AUD",  AUD,  "green"),
    ("SM",   SM,   "red"),
    ("PROD", PROD, "blue"),
]

SPEC_TYPE = "multitaper_smooth"
VLIM      = (0.8, 1.4)

# ---------------------------------------------------------------------------
# Representative electrodes for spectrogram rows
# Set each to "D{subj_num}-{channel_name}" or None to auto-select the first
# available electrode in that group (sorted by channel index).
# Examples: "D0005-LTG15", "D0016-MST1", "D0021-LST2"
# ---------------------------------------------------------------------------
SPEC_ELEC = {
    "AUD":  "D0064-LAI6",   # e.g. "D0005-LTG15"
    "SM":   "D0028-LPIO7",   # e.g. "D0016-MST1"
    "PROD": "D0022-LPIF4",   # e.g. "D0021-LST2"
}

# Matplotlib marker codes matched in brain insets and spectrogram y-labels.
SPEC_MARKERS_MPL = {"AUD": "^", "SM": "s", "PROD": "o"}
MARKER_SIZE = 8   # shared size for brain-inset and ylabel markers

# All three representative electrodes are in the SM group → mark them together
# on the SM brain inset.  Keys are group names; values are electrode names.
HIGHLIGHT_ELECS = dict(SPEC_ELEC)


def _get_elec_fsaverage_pos(spec_elec_str: str) -> np.ndarray | None:
    """Return fsaverage 3D position (metres) for a 'D0028-LPIO7' electrode."""
    subj_label, ch_name = spec_elec_str.split("-", 1)
    subj = f"D{int(subj_label[1:])}"   # 'D0028' → 'D28'
    subj_dir = get_sub_dir()
    try:
        info = subject_to_info(subj, subj_dir)
    except Exception:
        return None
    if ch_name not in info.ch_names:
        return None
    to_fsaverage = mne.read_talxfm(subj, subj_dir)
    trans = mne.transforms.Transform(
        fro='head', to='mri', trans=to_fsaverage['trans'])
    montage = info.get_montage()
    force2frame(montage, trans.from_str)
    montage.apply_trans(trans)
    return montage.get_positions()['ch_pos'].get(ch_name)


# ---------------------------------------------------------------------------
# Figure skeleton
# ---------------------------------------------------------------------------
fig = setup_figure()
gs_outer = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[1, 1],
    **GS_KWARGS,
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

# Areas: PROD = prod_only + sm_size, AUD = aud_only + sm_size
aud_total  = aud_only + sm_size
prod_total = prod_only + sm_size

# Radii area-proportional: PROD is the larger circle (normalised to R_BASE)
R_BASE = 1.
r_prod = R_BASE
r_aud  = R_BASE * np.sqrt(aud_total / prod_total)


def _circle_intersect_area(r1, r2, d):
    """Exact lens area of two circles with radii r1, r2 and centre distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    cos_a1 = np.clip((d**2 + r1**2 - r2**2) / (2 * d * r1), -1.0, 1.0)
    cos_a2 = np.clip((d**2 + r2**2 - r1**2) / (2 * d * r2), -1.0, 1.0)
    a1, a2 = np.arccos(cos_a1), np.arccos(cos_a2)
    sq = max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return r1**2 * a1 + r2**2 * a2 - 0.5 * np.sqrt(sq)


# Target overlap: sm_size/aud_total of AUD area  (= sm_size/prod_total of PROD area)
target_intersect = (sm_size / aud_total) * np.pi * r_aud**2

# Solve for exact centre distance
d_min = abs(r_aud - r_prod) + 1e-8
d_max = r_aud + r_prod - 1e-8
d_centers = brentq(
    lambda d: _circle_intersect_area(r_aud, r_prod, d) - target_intersect,
    d_min, d_max,
)

# Vertical layout: AUD on top, PROD on bottom
cy_aud  =  d_centers / 2
cy_prod = -d_centers / 2
cx = 0.0

# Tight axes limits — extra left margin to accommodate text labels
margin     = 0.0
lbl_margin = 0.0   # extra space on the left for the count labels
ax_venn.set_xlim(-(r_prod + lbl_margin), r_prod + margin)
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

# ax_venn.set_title("Electrode Groups", fontsize=11)

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
    (cx, y_aud_inset,  inset_hw, AUD,  "green", None),
    (cx, y_sm_inset,   inset_hw, SM,   "red",   HIGHLIGHT_ELECS),
    (cx, y_prod_inset, inset_hw, PROD, "blue",  None),
]

for icx, icy, hw, idx_set, color, highlight_elecs in inset_specs:
    shot, marker_px = _brain_screenshot(idx_set, color,
                                        highlight_elecs=highlight_elecs)
    h_img, w_img = shot.shape[:2]
    hh = hw * (h_img / w_img)
    ax_ins = ax_venn.inset_axes(_bounds_v(icx, icy, hw, hh), zorder=3)
    ax_ins.set_facecolor('none')
    ax_ins.imshow(shot)
    ax_ins.axis("off")
    for grp_key, px, py in marker_px:
        mpl_sym = SPEC_MARKERS_MPL.get(grp_key, "o")
        ax_ins.plot(px, py, marker=mpl_sym, linestyle='none',
                    markerfacecolor='red', markeredgecolor='black',
                    markeredgewidth=1, markersize=MARKER_SIZE, clip_on=False)

# Count labels — left of the diagram
lbl_x  = -(r_prod + 0.05)   # just left of the larger (PROD) circle
lbl_kw = dict(ha='right', va='center', zorder=6, fontsize=LABEL_SIZE)
ax_venn.text(lbl_x, y_aud_inset,  f"Auditory\n{aud_only}",      color='green', **lbl_kw)
ax_venn.text(lbl_x, y_sm_inset,   f"Sensory-Motor\n{sm_size}",  color='red',   **lbl_kw)
ax_venn.text(lbl_x, y_prod_inset, f"Production\n{prod_only}",   color='blue',  **lbl_kw)

# ============================================================
# RIGHT – 4 rows × 3 cols
#   Row 0       : frequency-averaged line plots (1 per condition)
#   Rows 1–3    : spectrograms — one row per group (AUD/SM/PROD),
#                 one col per condition — 3×3 total, single shared colorbar.
# Shared x-axis per column; x-labels only on the bottom row.
# ============================================================
gs_right = gridspec.GridSpecFromSubplotSpec(
    4, 4,
    subplot_spec=gs_outer[0, 1],
    height_ratios=[2, 1, 1, 1],
    width_ratios=[10, 10, 10, 1],
    hspace=GS_KWARGS['hspace'], wspace=GS_KWARGS['hspace'],
)

axes_ts   = []
axes_spec = [[None] * len(COND_LIST) for _ in range(len(GROUPS))]

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
                ha="center", va="center", transform=ax.transAxes, fontsize=TICK_SIZE)

    # ax.set_title(title, fontsize=TICK_SIZE)
    plt.setp(ax.get_xticklabels(), visible=False)

    if j == 0:
        ax.set_ylabel("HG Power (z)", fontsize=LABEL_SIZE)
        ax.legend(loc="upper left", fontsize=TICK_SIZE, framealpha=0.6)
        ylims_ts = ax.get_ylim()
    else:
        ax.set_yticklabels([])

if ylims_ts:
    for ax in axes_ts:
        ax.set_ylim(ylims_ts)

# ---- Peak-time boxplots on row 0 (same pattern as figure_4) ----
n_groups = len(GROUPS)
for j, (cond, timing, title) in enumerate(COND_LIST):
    ax = axes_ts[j]
    if cond not in avail_conds:
        continue
    arr = freq_avg(cond)
    yl = ax.get_ylim()
    positions = np.linspace(
        (yl[0] + yl[1]) * 4 / 5, yl[1], n_groups
    )
    width = positions[1] - positions[0]
    positions -= width / 2
    for i, (grp_name, grp_idx, color) in enumerate(GROUPS):
        idx_list = list(grp_idx)
        if not idx_list:
            continue
        grp_data = arr[idx_list]
        n_time = grp_data.shape[-1]
        ttimes = np.linspace(timing[0], timing[1], n_time)
        peak_times = ttimes[np.nanargmax(grp_data, axis=-1)]
        ax.boxplot(
            peak_times, vert=False, manage_ticks=False,
            positions=[positions[i]], widths=width / 2,
            patch_artist=True, boxprops=dict(facecolor=color),
            medianprops=dict(color="k", alpha=0.5), showfliers=False,
        )

# ---- Rows 1–3: spectrograms — 3 groups × 3 conditions ----
last_im = None

for i, (grp_name, grp_idx, color) in enumerate(GROUPS):
    if not grp_idx:
        for j in range(len(COND_LIST)):
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

    for j, (cond, timing, title) in enumerate(COND_LIST):
        ax = fig.add_subplot(gs_right[i + 1, j], sharex=axes_ts[j])
        axes_spec[i][j] = ax

        if i == len(GROUPS) - 1:
            ax.set_xlabel(f"Time from\n{title} (s)", fontsize=LABEL_SIZE)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        if j == 0:
            if i == 0:
                ax.set_ylabel("Freq (Hz)")
            mpl_sym = SPEC_MARKERS_MPL.get(grp_name, "o")
            ax.plot(-0.5, 0.5, marker=mpl_sym, transform=ax.transAxes,
                    clip_on=False, linestyle='none',
                    markerfacecolor='red', markeredgecolor='black',
                    markeredgewidth=0.75, markersize=MARKER_SIZE)
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
                    fontsize=TICK_SIZE, wrap=True)
            ax.axis("off")

# Single colorbar in the dedicated narrow 5th column (rows 1–3 only)
if last_im is not None:
    cax = fig.add_subplot(gs_right[1:, -1])
    cb = fig.colorbar(last_im, cax=cax, label=POWER_RATIO_LABEL)
    cb.set_label(POWER_RATIO_LABEL, fontsize=LABEL_SIZE)

# ---------------------------------------------------------------------------
# Subfigure labels
# ---------------------------------------------------------------------------
add_panel_label(ax_venn, "a", x=-0.05, y=1.02)
add_panel_label(axes_ts[0], "b", x=-0.15, y=1.02)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
finalize_figure(fig, "figure_2")
