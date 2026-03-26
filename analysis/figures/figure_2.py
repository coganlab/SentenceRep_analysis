"""Figure 2: Electrode group overview.

Layout:
  Left  (both rows) : Venn diagram — AUD (green) / PROD (blue) / SM overlap
                      (red) — with a left-hemisphere brain-render inset clipped
                      to each circle.
  Right top  (1×3)  : Frequency-averaged z-scored HG power for aud_ls, go_ls,
                      resp (AUD / SM / PROD coloured lines).
  Right bot  (1×3)  : Matching example spectrograms (one electrode per group),
                      sharing the x-axis with the row above.

Data source: stats_freq_hilbert combined LabeledArrays (see slice_freq.py).
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne.time_frequency
import numpy as np
from matplotlib.patches import Circle
from matplotlib_venn import venn2

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
    """Render group electrodes on fsaverage brain, return cropped RGB array.

    Mirrors the pattern in slice_freq.py / electrode_ratio_gradient:
    build the brain with show=False, copy actors into a BackgroundPlotter,
    then screenshot that plotter with return_img=True.
    """
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
    # Convert image to uint8 RGB(A) and build an alpha mask where background is
    # white (255). This makes non-brain (white) pixels transparent so the
    # underlying Venn diagram / axis content shows through when the inset is
    # drawn on top.
    if img.dtype != np.uint8:
        # normalize floats to 0-255
        try:
            img = (img * 255).astype(np.uint8)
        except Exception:
            img = img.astype(np.uint8)

    # Ensure we have at least RGB channels
    if img.ndim == 2:
        # grayscale -> RGB
        img = np.dstack([img, img, img])

    rgb = img[..., :3]

    # Non-white mask (any channel != 255)
    nw = (rgb != 255).any(-1)

    # Build alpha channel: opaque where any channel != 255, transparent otherwise
    alpha = np.where(nw, 255, 0).astype(np.uint8)

    # If original image had an alpha channel, combine conservatively
    if img.shape[-1] == 4:
        orig_a = img[..., 3].astype(np.uint8)
        new_a = np.maximum(orig_a, alpha)
        rgba = np.dstack((rgb, new_a))
    else:
        rgba = np.dstack((rgb, alpha))

    # Crop to content bbox (use the non-white mask computed above)
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

# One representative electrode per group (AUD col-0, SM col-1, PROD col-2)
GROUPS = [
    ("AUD",  AUD,  "green"),
    ("SM",   SM,   "red"),
    ("PROD", PROD, "blue"),
]

SPEC_TYPE = "multitaper_smooth"
VLIM      = (0.7, 1.4)

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
# LEFT – Venn diagram spanning the full left column
# ============================================================
ax_venn = fig.add_subplot(gs_outer[0, 0])

aud_only  = len(AUD  - SM - PROD)
prod_only = len(PROD - SM - AUD)
sm_size   = len(SM)

v = venn2(
    subsets=(aud_only, prod_only, sm_size),
    set_labels=("", ""),
    ax=ax_venn,
)

for pid, fc, txt in [("10", "green", f"AUD\nn={aud_only}"),
                      ("11", "red",   f"SM\nn={sm_size}"),
                      ("01", "blue",  f"PROD\nn={prod_only}")]:
    patch = v.get_patch_by_id(pid)
    lbl   = v.get_label_by_id(pid)
    if patch:
        patch.set_facecolor(fc)
        patch.set_alpha(0.08)
        patch.set_zorder(1)
    if lbl:
        lbl.set_text(txt)
        lbl.set_fontsize(9)
        lbl.set_zorder(3)

ax_venn.set_title("Electrode Groups", fontsize=11)

# Force draw so axes limits are set before we query them
fig.canvas.draw()

# -- extract circle geometry (Point2D → ndarray) --
def _pt(p):
    return np.array([float(p.x), float(p.y)])

c0 = _pt(v.centers[0])   # AUD circle centre
c1 = _pt(v.centers[1])   # PROD circle centre
r0, r1 = float(v.radii[0]), float(v.radii[1])
c_sm = (c0 + c1) / 2

direction = (c0 - c1) / np.linalg.norm(c0 - c1)   # unit vec pointing AUD-ward
aud_pos  = c0 + direction * r0 * 0.25
prod_pos = c1 - direction * r1 * 0.25

xlim, ylim = ax_venn.get_xlim(), ax_venn.get_ylim()
xr, yr = xlim[1] - xlim[0], ylim[1] - ylim[0]


def _bounds(cx, cy, hw, hh):
    """[x0, y0, w, h] in axes-fraction coordinates."""
    return [(cx - hw - xlim[0]) / xr,
            (cy - hh - ylim[0]) / yr,
            2 * hw / xr,
            2 * hh / yr]


# Render and place one brain inset per circle — sized to fill each region
for (cx, cy), half_r, idx_set, color in [
    (aud_pos,  r0 * 0.82, AUD,  "green"),
    (c_sm,     r0 * 0.70, SM,   "red"),
    (prod_pos, r1 * 0.82, PROD, "blue"),
]:
    shot = _brain_screenshot(idx_set, color)
    h_img, w_img = shot.shape[:2]
    hw = half_r
    hh = half_r * (h_img / w_img)

    ax_ins = ax_venn.inset_axes(_bounds(cx, cy, hw, hh), zorder=2)
    # Make inset axes fully transparent so RGBA image alpha shows the
    # underlying venn diagram through the image's transparent background.
    ax_ins.set_facecolor('none')
    im_brain = ax_ins.imshow(shot,
                             # interpolation="lanczos"
                             )
    ax_ins.axis("off")

    # clip_c = Circle((0.5, 0.5), 0.5, transform=ax_ins.transAxes)
    # ax_ins.add_patch(clip_c)
    # im_brain.set_clip_path(clip_c)

# ============================================================
# RIGHT – 4 rows × 3 cols
#   Row 0       : frequency-averaged line plots (1 per condition)
#   Rows 1–3    : spectrograms — one row per group (AUD/SM/PROD),
#                 one col per condition — 3×3 total.
# Shared x-axis per column; x-labels only on the bottom row.
# ============================================================
gs_right = gridspec.GridSpecFromSubplotSpec(
    4, 3,
    subplot_spec=gs_outer[0, 1],
    height_ratios=[2, 1, 1, 1],
    hspace=0.10,
    wspace=0.30,
)

axes_ts   = []
axes_spec = []

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
    plt.setp(ax.get_xticklabels(), visible=False)   # hide – shared with row below

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
# axes_spec[i][j] = row i (group), col j (condition)
axes_spec = [[None] * 3 for _ in range(3)]

for i, (grp_name, grp_idx, color) in enumerate(GROUPS):
    if not grp_idx:
        for j in range(3):
            ax = fig.add_subplot(gs_right[i + 1, j])
            ax.text(0.5, 0.5, "no electrodes", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            axes_spec[i][j] = ax
        continue

    ch_i    = sorted(grp_idx)[0]
    ch_full = all_chans[ch_i]
    subj, ch_name = ch_full.split("-", 1)

    for j, (cond, timing, _title) in enumerate(COND_LIST):
        ax = fig.add_subplot(gs_right[i + 1, j], sharex=axes_ts[j])
        axes_spec[i][j] = ax

        # x-label only on the bottom spectrogram row
        if i == len(GROUPS) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        # y-label / ticks: only leftmost column
        if j == 0:
            ax.set_ylabel(f"{grp_name}\nFreq (Hz)", color=color, fontsize=8)
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
            # colorbar only on rightmost column
            if j == 2:
                plt.colorbar(im, ax=ax, label="Power ratio", pad=0.02)
            ax.set_xlim(timing)

        except (OSError, ValueError) as exc:
            ax.text(0.5, 0.5, f"{ch_full}\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, wrap=True)
            ax.axis("off")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.suptitle("Figure 2", fontsize=14, y=0.995)
fig.savefig(os.path.join(out_dir, "figure_2.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_2.png"), bbox_inches="tight", dpi=150)
plt.show()