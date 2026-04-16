"""Figure 5: Component-Weighted Word Decoding Performance.

Layout (1 x 3):
  Component-weighted word decoding (Auditory / WM / Motor / Visual SliceTCA
  components) for three alignment conditions (aud, go, resp), with faded
  Sensory-Motor electrode-group decoding overlaid for reference.
  Data source: words_factor.py and load_freq.py pre-computed scores.
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
    ANALYSIS_DIR, DECOMPOSITION_DIR, DECODING_DIR,
)
from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist, plot_dist_bound
from analysis.utils.plotting import plot_horizontal_bars

# ---------------------------------------------------------------------------
# Paths to pre-computed score files
# ---------------------------------------------------------------------------
# Electrode-group scores (for SM reference trace)
SM_TRUE = os.path.join(DECOMPOSITION_DIR,
                       "true_scores_zscore_nofreqmult_word_AUDSMPROD4.npz")
SM_SHUF = os.path.join(DECOMPOSITION_DIR,
                       "shuffle_scores_zscore_nofreqmult_word_AUDSMPROD4.npz")

# Component-weighted scores
BOT_TRUE = os.path.join(DECODING_DIR,
                        "true_scores_zscore_weighted_words29.npz")
BOT_SHUF = os.path.join(DECODING_DIR,
                        "shuffle_scores_zscore_weighted_words29.npz")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES = 4
BASELINE = 1 / N_CLASSES
YLIMS = (BASELINE - 0.3, BASELINE + 0.6)

CONDS = [["aud_ls", "aud_lm"], ["go_ls", "go_lm"], "resp"]
COND_TITLES = {
    "aud_ls-aud_lm": "Stimulus Onset",
    "go_ls-go_lm":   "Go Cue",
    "resp":           "Response",
}

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
    if cond_str == "resp":
        return (-0.9, 0.9)
    return (-0.4, 1.4)


def _times_shuffle_comp(cond_str: str) -> tuple[float, float]:
    if cond_str.endswith("resp"):
        return (-1, 1)
    return (-0.5, 1.25)


def _times_shuffle_sm(cond_str: str) -> tuple[float, float]:
    if cond_str.endswith("resp"):
        return (-1, 1)
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
fig = setup_figure(figsize=(18 * cm, 7.5 * cm))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.15)

axes = []
for j, cond in enumerate(CONDS):
    ax = fig.add_subplot(gs[0, j])
    axes.append(ax)

    cond_str = "-".join(cond) if isinstance(cond, list) else cond
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
                        _times_shuffle_sm(cond_str), 0,
                        ax=ax, color='grey', alpha=0.15, linewidth=0)

    # Component shuffles
    for name, color in COMP_SERIES:
        key = _cond_key(name, cond)
        if key not in bot_shuf:
            continue
        true_acc = _acc(bot_true[key])
        shuf_acc = _acc(bot_shuf[key])
        # sig = time_perm_cluster(
        #     true_acc.mean(axis=1, keepdims=True).T,
        #     shuf_acc.T, 0.05, n_perm=10000,
        #     stat_func=_stat,
        # )[0]
        sig = time_perm_cluster(#true_acc.T,
            true_acc.mean(axis=1, keepdims=True).T,
            shuf_acc.T, 0.08, n_perm=10000,
            stat_func=lambda x, y, axis: np.mean(x, axis=axis)
        )[0]
        bars.append(sig)
        bar_colors.append(color)
        bar_times.append(times_t)
        window = np.lib.stride_tricks.sliding_window_view(
            shuf_acc, 20, axis=0)
        shuf_smooth = np.mean(window, axis=-1)
        plot_dist_bound(shuf_smooth, "std", "both",
                        _times_shuffle_comp(cond_str), 0,
                        ax=ax, color='grey', alpha=0.2, linewidth=0)

    if bars:
        plot_horizontal_bars(ax, bars, 0.02, "below",
                             colors=bar_colors, times=bar_times)

    ax.axhline(BASELINE, color="k", linestyle="--", linewidth=0.5)
    ax.set_ylim(*YLIMS)
    if cond_str == "resp":
        ax.set_xlabel("Time from response (s)", fontsize=LABEL_SIZE)
    elif "aud" in cond_str:
        ax.set_xlabel("Time from stimulus (s)", fontsize=LABEL_SIZE)
    else:
        ax.set_xlabel("Time from go cue (s)", fontsize=LABEL_SIZE)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x * 100:.0f}"))
    if j == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper left", framealpha=0.6)
    else:
        ax.set_yticklabels([])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_5.svg"), bbox_inches="tight", dpi=DPI)
fig.savefig(os.path.join(out_dir, "figure_5.png"), bbox_inches="tight", dpi=DPI)
plt.show()
