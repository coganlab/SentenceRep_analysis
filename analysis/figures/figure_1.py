"""Figure 1: Task design and example high-gamma spectrograms.

Layout (per condition row: pac-man icon above a [monitor | spectrogram] pair;
three rows JL / LM / LS stacked vertically; two such columns side by side for
the stimulus and go-cue epochs).  The colour bar sits to the right of the
middle-row right spectrogram.

The two vertical dashed lines are drawn once as figure-level artists and span
all three spectrogram rows:
  • Left dashed line  : stim onset (t = 0) on the left spectrograms.
  • Right dashed line : response onset (t = RESP_LAT = 0.873 s) on the right
                         spectrograms (right specs are go-cue aligned).

The timeline at the bottom is also drawn in figure fractions so every epoch
boundary lands exactly under the corresponding x-position above:
  • Start Cue : left edge of the monitor column  → t = 0 on left spec
  • Stim      : t = 0 → t = 0.5   on left spec
  • Delay     : t = 0.5 on left spec → t = 0 on right spec  (spans the gap)
  • Go Cue    : t = 0 → t = RESP_LAT on right spec
  • Arrow     : continues past RESP_LAT; "Response Onset" labels RESP_LAT.

Spectrogram sources (replicates `analysis/check/single_check.py`):
  Left column  : subject 24 (D0024), pick = 'LTG7'   — aud_jl / aud_lm / aud_ls
  Right column : subject 17 (D0017), pick = 'RPIF8'  — go_jl  / go_lm  / go_ls

Icon assets in `analysis/figures/4x/`:
  Asset 1@4x.png   — blank monitor
  Asset 2@4x.png   — "speaking" pac-man (face + empty speech bubble)
  Asset 3@4x.png   — "closed-mouth" pac-man
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mne.time_frequency
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE, DPI, LAB_ROOT, POWER_RATIO_LABEL,
    finalize_figure, add_panel_label,
)
from ieeg.calc.scaling import rescale
from ieeg.io import get_data
from ieeg.viz.parula import parula_map

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK = "SentenceRep"
SPEC_TYPE = "multitaper_smooth"
VLIM = (0.7, 1.4)
FMIN, FMAX = 0, 250

LEFT_SUBJECT, LEFT_PICK = 24, "LTG7"        # stimulus-aligned specs
RIGHT_SUBJECT, RIGHT_PICK = 17, "RPIF8"     # go-cue-aligned specs

# Trial timing (s)
STIM_DUR = 0.5
RESP_LAT = 0.873       # response onset relative to go-cue onset

ROW_SPECS = [
    ("Just Listen\n[JL]",  "aud_jl", "go_jl", "jl"),
    ("Listen Mime\n[LM]",  "aud_lm", "go_lm", "lm"),
    ("Listen Speak\n[LS]", "aud_ls", "go_ls", "ls"),
]

# Icon assets
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(FIGURES_DIR, "4x")
BASE_ICONS = {
    "monitor":       "Asset 1@4x.png",
    "pacman_speak":  "Asset 2@4x.png",
    "pacman_closed": "Asset 3@4x.png",
}
# (row_key, col_key) -> (monitor_text, pacman_variant, bubble_text)
CELL_ICONS = {
    ("jl", "aud"): (":=:",    "pacman_speak",  ""),
    ("lm", "aud"): ("Listen", "pacman_speak",  ""),
    ("ls", "aud"): ("Listen", "pacman_speak",  "h_t"),
    ("jl", "go"):  ("",       "pacman_closed", ""),
    ("lm", "go"):  ("Mime",   "pacman_speak",  ""),
    ("ls", "go"):  ("Speak",  "pacman_speak",  "h_t"),
}
# Overlay positions in axes fractions — tune once the first render is inspected
MONITOR_TEXT_XY = (0.5, 0.45)
BUBBLE_TEXT_XY  = (0.72, 0.72)

# Column widths (repeated across all rows)
#  0: row label | 1: monitor_L | 2: spec_L | 3: gap
#  4: monitor_R | 5: spec_R    | 6: cbar
WIDTH_RATIOS = [0.35, 0.5, 1.8, 0.2, 0.5, 1.8, 0.08]


# ---------------------------------------------------------------------------
# Data loading (mirrors single_check.py)
# ---------------------------------------------------------------------------
def _resolve_subj(layout, subject_int):
    for subj in layout.get(return_type="id", target="subject"):
        if int(subj[1:]) == subject_int:
            return subj
    raise ValueError(f"Subject {subject_int} not found under {layout.root}")


def load_spec_data(layout, subject, pick, cond):
    subj_id = _resolve_subj(layout, subject)
    base = (mne.time_frequency.read_tfrs(os.path.join(
                layout.root, "derivatives", "spec", SPEC_TYPE,
                subj_id, "start-tfr.h5"))
            .average(lambda x: np.nanmean(x, axis=0))
            .crop(tmin=-0.5, tmax=0))
    spec_a = (mne.time_frequency.read_tfrs(os.path.join(
                layout.root, "derivatives", "spec", SPEC_TYPE,
                subj_id, f"{cond}-tfr.h5"))
              .average(lambda x: np.nanmean(x, axis=0)))
    rescale(spec_a, base, mode="ratio", copy=False)
    ch_idx = spec_a.ch_names.index(pick)
    data = spec_a.data[ch_idx]
    fmask = (spec_a.freqs >= FMIN) & (spec_a.freqs <= FMAX)
    return data[fmask], spec_a.freqs[fmask], spec_a.times


# ---------------------------------------------------------------------------
# Icon helpers — each icon gets its own axes so bubble/screen text can use
# axes fractions of that axes directly.
# ---------------------------------------------------------------------------
def _load_base(key):
    fn = BASE_ICONS.get(key)
    if fn is None:
        return None
    path = os.path.join(ICON_DIR, fn)
    if not os.path.exists(path):
        return None
    return mpimg.imread(path)


def _draw_icon_axes(ax, img, text=None, text_xy=(0.5, 0.5),
                    text_color="black", text_weight="bold"):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if img is not None:
        ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto",
                  interpolation="bilinear")
    if text:
        ax.text(*text_xy, text, ha="center", va="center",
                transform=ax.transAxes, fontsize=TICK_SIZE,
                color=text_color, fontweight=text_weight, zorder=20)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
layout = get_data(TASK, root=LAB_ROOT)
monitor_img = _load_base("monitor")
pacman_speak_img = _load_base("pacman_speak")
pacman_closed_img = _load_base("pacman_closed")

fig = setup_figure(figsize=(22 * cm, 13 * cm))

# Outer: data (top) + timeline whitespace (bottom) where figure-level rects live
gs_outer = gridspec.GridSpec(
    2, 1, figure=fig, height_ratios=[1, 0.14], hspace=0.04,
)

# Three condition rows, each with [pac-man row, content row]
gs_data = gs_outer[0].subgridspec(3, 1, hspace=0.3)

# Trackers
top_left_spec = bot_left_spec = top_right_spec = bot_right_spec = None
first_mon_left_ax = None
cax = None
im = None

for row, (row_label, aud_cond, go_cond, row_key) in enumerate(ROW_SPECS):
    row_gs = gs_data[row].subgridspec(
        2, 7,
        height_ratios=[0.35, 1],
        width_ratios=WIDTH_RATIOS,
        hspace=0.02, wspace=0.12,
    )

    # Row label spans the pac-man + content sub-rows
    ax_lbl = fig.add_subplot(row_gs[:, 0])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.5, row_label, ha="center", va="center",
                fontsize=LABEL_SIZE)

    for cond, col_key, subject, pick, mon_col, spec_col in [
        (aud_cond, "aud", LEFT_SUBJECT,  LEFT_PICK,  1, 2),
        (go_cond,  "go",  RIGHT_SUBJECT, RIGHT_PICK, 4, 5),
    ]:
        mon_text, pac_key, bubble_text = CELL_ICONS[(row_key, col_key)]

        # --- Pac-man / speaker icon (upper sub-row) ---
        ax_pac = fig.add_subplot(row_gs[0, spec_col])
        pac_img = (pacman_speak_img if pac_key == "pacman_speak"
                   else pacman_closed_img)
        _draw_icon_axes(ax_pac, pac_img, text=bubble_text or None,
                        text_xy=BUBBLE_TEXT_XY)

        # --- Monitor icon (content sub-row, monitor column) ---
        ax_mon = fig.add_subplot(row_gs[1, mon_col])
        _draw_icon_axes(ax_mon, monitor_img, text=mon_text or None,
                        text_xy=MONITOR_TEXT_XY, text_color="white")

        # --- Spectrogram (content sub-row, spec column) ---
        ax = fig.add_subplot(row_gs[1, spec_col])
        data, freqs, times = load_spec_data(layout, subject, pick, cond)
        im = ax.imshow(
            data, aspect="auto", origin="lower",
            cmap=parula_map, vmin=VLIM[0], vmax=VLIM[1],
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        ax.set_xticks([])
        if col_key == "go":
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelsize=TICK_SIZE)
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
            ax.set_yticks([0, 250])
            if row == 1:
                ax.set_ylabel("Frequency (Hz)", fontsize=LABEL_SIZE)
        else:
            ax.set_yticks([])

        # Track the four corner specs + first monitor for later positioning
        if col_key == "aud":
            if row == 0:
                top_left_spec = ax
                first_mon_left_ax = ax_mon
            if row == len(ROW_SPECS) - 1:
                bot_left_spec = ax
        else:
            if row == 0:
                top_right_spec = ax
            if row == len(ROW_SPECS) - 1:
                bot_right_spec = ax

        # Colorbar axes only on middle row, right of right spec
        if row == 1 and col_key == "go":
            cax = fig.add_subplot(row_gs[1, 6])

# Single colour bar (middle row only)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(POWER_RATIO_LABEL, fontsize=LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_SIZE)
cbar.set_ticks([VLIM[0], VLIM[1]])

# ---------------------------------------------------------------------------
# Figure-level overlays (dashed lines + timeline)
# Use gridspec positions + transData so every x-coord aligns exactly with the
# axes above.
# ---------------------------------------------------------------------------
fig.canvas.draw()  # force layout so transforms return real positions


def _fig_x(ax, data_x):
    """Convert an axis data-x value into figure-fraction x."""
    disp = ax.transData.transform((data_x, 0))
    return fig.transFigure.inverted().transform(disp)[0]


def _fig_bbox(ax):
    """Axes data-region bbox in figure-fraction coordinates."""
    return ax.get_position()


# Key x-positions (figure fractions)
x_stim_on  = _fig_x(top_left_spec,  0.0)
x_stim_off = _fig_x(top_left_spec,  STIM_DUR)
x_go_on    = _fig_x(top_right_spec, 0.0)
x_resp_on  = _fig_x(top_right_spec, RESP_LAT)

mon_bbox       = _fig_bbox(first_mon_left_ax)
top_L_bbox     = _fig_bbox(top_left_spec)
bot_L_bbox     = _fig_bbox(bot_left_spec)
top_R_bbox     = _fig_bbox(top_right_spec)
bot_R_bbox     = _fig_bbox(bot_right_spec)

# --- Dashed vertical lines spanning all three spec rows ---
for x_fig, y0, y1 in [
    (x_stim_on, bot_L_bbox.y0, top_L_bbox.y1),
    (x_resp_on, bot_R_bbox.y0, top_R_bbox.y1),
]:
    fig.add_artist(Line2D(
        [x_fig, x_fig], [y0, y1],
        transform=fig.transFigure,
        color="k", linestyle="--", linewidth=1.2, zorder=50,
    ))

# --- Timeline bar ---
tl_pos = gs_outer[1].get_position(fig)
y_bar = tl_pos.y0 + (tl_pos.y1 - tl_pos.y0) * 0.55
bar_h = (tl_pos.y1 - tl_pos.y0) * 0.30

segments = [
    ("Start Cue (0.5s)", mon_bbox.x0,  x_stim_on,   "black"),
    ("Stim (0.5s)",      x_stim_on,    x_stim_off,  "grey"),
    ("Delay (0.75s)",    x_stim_off,   x_go_on,     "black"),
    ("Go Cue (0.5s)",    x_go_on,      x_resp_on,   "grey"),
]
for label, x0, x1, col in segments:
    fig.add_artist(Rectangle(
        (x0, y_bar), x1 - x0, bar_h,
        facecolor=col, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=5,
    ))
    fig.text((x0 + x1) / 2, y_bar - bar_h * 0.5, label,
             ha="center", va="top", fontsize=TICK_SIZE)

# Response Onset label at the dashed line position
fig.text(x_resp_on + 0.002, y_bar + bar_h + 0.005, "Response Onset",
         ha="left", va="bottom", fontsize=TICK_SIZE)

# Arrow past response onset, extending to the right edge of the right spec
arrow_x_end = top_R_bbox.x1 + 0.015
fig.add_artist(FancyArrowPatch(
    (x_resp_on, y_bar + bar_h / 2),
    (arrow_x_end, y_bar + bar_h / 2),
    arrowstyle="-|>", linewidth=1.2, color="black",
    mutation_scale=12, transform=fig.transFigure, clip_on=False,
))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
finalize_figure(fig, "figure_1")
