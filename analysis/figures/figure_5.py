"""Figure 5: Word Decoding Performance.

Layout (2 x 3):
  Top row    : Electrode-group word decoding (Production / Sensory-Motor /
               Auditory) for three alignment conditions (aud, go, resp).
               Data source: load_freq.py pre-computed scores.
  Bottom row : Component-weighted word decoding (Auditory / WM / Motor /
               Visual SliceTCA components) for the same three conditions.
               Data source: words_factor.py pre-computed scores.
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures.config import cm, GS_KWARGS, setup_figure, LABEL_SIZE, TICK_SIZE
from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist, plot_dist_bound
from analysis.utils.plotting import plot_horizontal_bars

# ---------------------------------------------------------------------------
# Paths to pre-computed score files
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOP_TRUE = os.path.join(REPO, "decomposition",
                        "true_scores_zscore_nofreqmult_word_AUDSMPROD4.npz")
TOP_SHUF = os.path.join(REPO, "decomposition",
                        "shuffle_scores_zscore_nofreqmult_word_AUDSMPROD4.npz")

BOT_TRUE = os.path.join(REPO, "decoding",
                        "true_scores_zscore_weighted_words29.npz")
BOT_SHUF = os.path.join(REPO, "decoding",
                        "shuffle_scores_zscore_weighted_words29.npz")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES = 4
BASELINE = 1 / N_CLASSES
YLIMS = (BASELINE - 0.2, BASELINE + 0.6)

CONDS = [["aud_ls", "aud_lm"], ["go_ls", "go_lm"], "resp"]
COND_TITLES = {
    "aud_ls-aud_lm": "Stimulus Onset",
    "go_ls-go_lm":   "Go Cue",
    "resp":           "Response",
}

# Top row: electrode-group decoding (load_freq.py)
TOP_SERIES = [
    ("Production",    [0, 0, 1]),
    ("Sensory-Motor", [1, 0, 0]),
    ("Auditory",      [0, 1, 0]),
]

# Bottom row: component-weighted decoding (words_factor.py)
BOT_SERIES = [
    ("Auditory", "orange"),
    ("WM",       "k"),
    ("Motor",    "c"),
    ("Visual",   "y"),
]

# Styling (matches figure_2 / figure_4)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _acc(cm_arr: np.ndarray) -> np.ndarray:
    """Extract mean diagonal accuracy from (..., C, C) confusion matrices.

    Parameters
    ----------
    cm_arr : np.ndarray, shape (time, folds, C, C)

    Returns
    -------
    np.ndarray, shape (time, folds)
    """
    eye = np.eye(N_CLASSES, dtype=bool)
    return np.mean(cm_arr.T[eye].T, axis=2)


def _cond_key(name: str, cond) -> str:
    """Build the dict key used in the .npz files."""
    if isinstance(cond, list):
        return f"{name}-{'-'.join(cond)}"
    return f"{name}-{cond}"


def _times_true(cond_str: str) -> tuple[float, float]:
    """Time range for true-score traces (matches plot_all_scores)."""
    if cond_str == "resp":
        return (-0.9, 0.9)
    return (-0.4, 1.4)


def _times_shuffle(cond_str: str, row: str) -> tuple[float, float]:
    """Time range for shuffle overlay."""
    if cond_str.endswith("resp"):
        return (-1, 1)
    if row == "top":
        return (-0.5, 1.5)
    return (-0.5, 1.25)


# ---------------------------------------------------------------------------
# Load data (minimal: four .npz files only)
# ---------------------------------------------------------------------------
top_true = dict(np.load(TOP_TRUE, allow_pickle=True))
top_shuf = dict(np.load(TOP_SHUF, allow_pickle=True))
bot_true = dict(np.load(BOT_TRUE, allow_pickle=True))
bot_shuf = dict(np.load(BOT_SHUF, allow_pickle=True))


# ---------------------------------------------------------------------------
# Figure skeleton
# ---------------------------------------------------------------------------
fig = setup_figure()
gs = gridspec.GridSpec(2, 3, figure=fig, **GS_KWARGS)


def _plot_row(row_idx: int, true_scores: dict, shuf_scores: dict,
              series: list[tuple], row_tag: str,
              stat_func: callable) -> list[plt.Axes]:
    """Populate one row (3 panels) of decoding results.

    Each panel shows:
      - true-score distribution (mean +/- std via plot_dist)
      - smoothed shuffle distribution (fill_between via plot_dist_bound)
      - cluster-corrected significance bars (plot_horizontal_bars)
      - chance-level dashed line
    """
    axes = []
    for j, cond in enumerate(CONDS):
        ax = fig.add_subplot(gs[row_idx, j])
        axes.append(ax)

        cond_str = "-".join(cond) if isinstance(cond, list) else cond
        times_t = _times_true(cond_str)

        # ---- True-score traces ----
        for name, color in series:
            key = _cond_key(name, cond)
            if key not in true_scores:
                continue
            acc = _acc(true_scores[key])                       # (T, folds)
            pl = acc.reshape(acc.shape[0], -1).T               # (folds, T)
            plot_dist(pl, mode="std", times=times_t,
                      color=color, label=name, ax=ax)

        # ---- Shuffle overlay + significance bars ----
        bars = []
        bar_colors = []
        bar_times = []
        for name, color in series:
            key = _cond_key(name, cond)
            if key not in shuf_scores:
                continue

            true_acc = _acc(true_scores[key])                  # (T, folds)
            shuf_acc = _acc(shuf_scores[key])                  # (T, reps)

            # Permutation-cluster significance
            sig = time_perm_cluster(
                true_acc.mean(axis=1, keepdims=True).T,
                shuf_acc.T, 0.05, n_perm=10000,
                stat_func=stat_func,
            )[0]
            bars.append(sig)
            bar_colors.append(color)
            bar_times.append(times_t)

            # Smooth shuffle with sliding window (window = 20, same as decode)
            window = np.lib.stride_tricks.sliding_window_view(
                shuf_acc, 20, axis=0)
            shuf_smooth = np.mean(window, axis=-1)
            times_sh = _times_shuffle(cond_str, row_tag)
            plot_dist_bound(shuf_smooth, "std", "both", times_sh, 0,
                            ax=ax, color=color, alpha=0.3)

        if bars:
            plot_horizontal_bars(ax, bars, 0.02, "below",
                                 colors=bar_colors, times=bar_times)

        # ---- Chance level ----
        ax.axhline(BASELINE, color="k", linestyle="--", linewidth=0.5)

        # ---- Formatting ----
        ax.set_ylim(*YLIMS)
        ax.tick_params(labelsize=TICK_SIZE)

        if row_idx == 0:
            ax.set_title(COND_TITLES[cond_str], fontsize=LABEL_SIZE)
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            if cond_str == "resp":
                ax.set_xlabel("Time from response (s)", fontsize=LABEL_SIZE)
            elif "aud" in cond_str:
                ax.set_xlabel("Time from stimulus (s)", fontsize=LABEL_SIZE)
            else:
                ax.set_xlabel("Time from go cue (s)", fontsize=LABEL_SIZE)

        if j == 0:
            ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
            ax.legend(fontsize=TICK_SIZE, loc="upper left",
                      framealpha=0.6)
        else:
            ax.set_yticklabels([])

    return axes


# ---------------------------------------------------------------------------
# Top row — electrode-group decoding (load_freq.py)
# ---------------------------------------------------------------------------
_stat_top = lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)
axes_top = _plot_row(0, top_true, top_shuf, TOP_SERIES, "top", _stat_top)

# ---------------------------------------------------------------------------
# Bottom row — component-weighted decoding (words_factor.py)
# ---------------------------------------------------------------------------
_stat_bot = lambda x, y, axis: np.mean(x, axis=axis)
axes_bot = _plot_row(1, bot_true, bot_shuf, BOT_SERIES, "bottom", _stat_bot)

# ---------------------------------------------------------------------------
# Subfigure labels
# ---------------------------------------------------------------------------
axes_top[0].text(-0.15, 1.1, "a", transform=axes_top[0].transAxes,
                 fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")
axes_bot[0].text(-0.15, 1.1, "b", transform=axes_bot[0].transAxes,
                 fontsize=LABEL_SIZE + 2, fontweight="bold", va="bottom")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_5.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_5.png"), bbox_inches="tight", dpi=300)
plt.show()
