"""Figure 6: Component-Weighted Pairwise Condition Decoding Performance.

Layout (3 x 2):
  Rows correspond to pairwise contrasts (LS vs LM, LS vs JL, LM vs JL).
  Columns correspond to epoch (Stimulus Onset, Go Cue).
  Each panel shows component-weighted 2-way condition decoding with faded
  Sensory-Motor electrode-group decoding overlaid for reference.
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
import numpy as np

from analysis.figures.config import (
    cm, GS_KWARGS, setup_figure, LABEL_SIZE, TICK_SIZE, DPI,
    COMP_NAMES, COMP_COLORS_LIST,
    DECOMPOSITION_DIR, DECODING_DIR,
)
from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist, plot_dist_bound
from analysis.utils.plotting import plot_horizontal_bars

# ---------------------------------------------------------------------------
# Paths to pre-computed score files
# ---------------------------------------------------------------------------
# Electrode-group scores (for SM reference trace)
SM_TRUE = os.path.join(DECOMPOSITION_DIR,
                       "true_scores_zscore_nofreqmult_2way_AUDSMPROD.npz")
SM_SHUF = os.path.join(DECOMPOSITION_DIR,
                       "shuffle_scores_zscore_nofreqmult_2way_AUDSMPROD.npz")

# Component-weighted scores
BOT_TRUE = os.path.join(DECODING_DIR,
                        "true_scores_zscore_weighted_words_2way3.npz")
BOT_SHUF = os.path.join(DECODING_DIR,
                        "shuffle_scores_zscore_weighted_words_2way3.npz")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES = 2
BASELINE = 1 / N_CLASSES   # 0.5
YLIMS = (BASELINE - 0.3, BASELINE + 0.5)

# Rows = contrasts, Columns = epochs (stimulus, go)
CONTRASTS = [
    (["aud_ls", "aud_lm"], ["go_ls", "go_lm"]),   # LS vs LM
    (["aud_ls", "aud_jl"], ["go_ls", "go_jl"]),    # LS vs JL
    (["aud_lm", "aud_jl"], ["go_lm", "go_jl"]),   # LM vs JL
]
ROW_LABELS = ["LS vs LM", "LS vs JL", "LM vs JL"]
COL_LABELS = ["Time from stimulus (s)", "Time from go cue (s)"]

# Component series (full opacity)
COMP_SERIES = list(zip(COMP_NAMES, COMP_COLORS_LIST))

# SM reference series (faded)
SM_NAME = "Sensory-Motor"
SM_COLOR = "#994444"
SM_ALPHA = 0.35
SM_BAR_COLOR = tuple(c * SM_ALPHA + (1 - SM_ALPHA) for c in to_rgb(SM_COLOR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _acc(cm_arr: np.ndarray) -> np.ndarray:
    eye = np.eye(N_CLASSES, dtype=bool)
    return np.mean(cm_arr.T[eye].T, axis=2)


def _cond_key(name: str, cond) -> str:
    if isinstance(cond, list):
        return f"{name}-{'-'.join(cond)}"
    return f"{name}-{cond}"


def _times_true(cond_str: str) -> tuple[float, float]:
    return (-0.4, 1.4)


def _times_shuffle(cond_str: str) -> tuple[float, float]:
    return (-0.5, 1.5)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sm_true = dict(np.load(SM_TRUE, allow_pickle=True))
sm_shuf = dict(np.load(SM_SHUF, allow_pickle=True))
bot_true = dict(np.load(BOT_TRUE, allow_pickle=True))
bot_shuf = dict(np.load(BOT_SHUF, allow_pickle=True))

_stat = lambda x, y, axis: np.mean(x, axis=axis)
_stat_sm = lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(12 * cm, 14 * cm))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.15)

axes = []
for row, (aud_cond, go_cond) in enumerate(CONTRASTS):
    for col, cond in enumerate([aud_cond, go_cond]):
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        cond_str = "-".join(cond)
        times_t = _times_true(cond_str)

        # ---- SM reference trace (faded, behind) ----
        sm_key = _cond_key(SM_NAME, cond)
        if sm_key in sm_true:
            acc = _acc(sm_true[sm_key])
            pl = acc.reshape(acc.shape[0], -1).T
            plot_dist(pl, mode="std", times=times_t,
                      color=SM_COLOR, label=SM_NAME, ax=ax, alpha=SM_ALPHA)

        # ---- Component traces (full opacity) ----
        for name, color in COMP_SERIES:
            key = _cond_key(name, cond)
            if key not in bot_true:
                continue
            acc = _acc(bot_true[key])
            pl = acc.reshape(acc.shape[0], -1).T
            plot_dist(pl, mode="std", times=times_t,
                      color=color, label=name, ax=ax)

        # ---- Shuffle overlay + significance bars ----
        bars = []
        bar_colors = []
        bar_times = []

        # SM shuffle
        if sm_key in sm_shuf:
            true_acc = _acc(sm_true[sm_key])
            shuf_acc = _acc(sm_shuf[sm_key])
            sig = time_perm_cluster(
                true_acc.mean(axis=1, keepdims=True).T,
                shuf_acc.T, 0.05, n_perm=10000,
                stat_func=_stat_sm,
            )[0]
            bars.append(sig)
            bar_colors.append(SM_BAR_COLOR)
            bar_times.append(times_t)
            window = np.lib.stride_tricks.sliding_window_view(
                shuf_acc, 20, axis=0)
            shuf_smooth = np.mean(window, axis=-1)
            plot_dist_bound(shuf_smooth, "std", "both",
                            _times_shuffle(cond_str), 0,
                            ax=ax, color='grey', alpha=0.15, linewidth=0)

        # Component shuffles
        for name, color in COMP_SERIES:
            key = _cond_key(name, cond)
            if key not in bot_shuf:
                continue
            true_acc = _acc(bot_true[key])
            shuf_acc = _acc(bot_shuf[key])
            sig = time_perm_cluster(
                true_acc.mean(axis=1, keepdims=True).T,
                shuf_acc.T, 0.05, n_perm=10000,
                stat_func=_stat,
            )[0]
            bars.append(sig)
            bar_colors.append(color)
            bar_times.append(times_t)
            window = np.lib.stride_tricks.sliding_window_view(
                shuf_acc, 20, axis=0)
            shuf_smooth = np.mean(window, axis=-1)
            plot_dist_bound(shuf_smooth, "std", "both",
                            _times_shuffle(cond_str), 0,
                            ax=ax, color='grey', alpha=0.2, linewidth=0)

        if bars:
            plot_horizontal_bars(ax, bars, 0.02, "below",
                                 colors=bar_colors, times=bar_times)

        ax.axhline(BASELINE, color="k", linestyle="--", linewidth=0.5)
        ax.set_ylim(*YLIMS)

        # x-labels only on bottom row
        if row == len(CONTRASTS) - 1:
            ax.set_xlabel(COL_LABELS[col], fontsize=LABEL_SIZE)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        # y-label and legend only on leftmost column, top row
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x * 100:.0f}"))
        if col == 0 and row == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
            ax.legend(fontsize=TICK_SIZE, loc="upper left",
                      framealpha=0.6)
        elif col > 0:
            ax.set_yticklabels([])

        # Row label on right side
        if col == 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(ROW_LABELS[row], fontsize=LABEL_SIZE,
                          rotation=270, labelpad=12)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_6.svg"), bbox_inches="tight", dpi=DPI)
fig.savefig(os.path.join(out_dir, "figure_6.png"), bbox_inches="tight", dpi=DPI)
plt.show()
