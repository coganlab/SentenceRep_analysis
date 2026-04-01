"""Figure 6: Pairwise Condition Decoding Performance.

Layout (2 x 6):
  Top row    : Electrode-group 2-way condition decoding (Production /
               Sensory-Motor / Auditory) for six pairwise comparisons.
               Data source: load_freq.py pre-computed scores.
  Bottom row : Component-weighted 2-way condition decoding (Auditory / WM /
               Motor / Visual SliceTCA components) for the same six
               pairwise comparisons.
               Data source: words_factor.py pre-computed scores.

Columns (6 pairwise condition comparisons):
  aud: LS vs LM, LS vs JL, LM vs JL
  go : LS vs LM, LS vs JL, LM vs JL
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
                        "true_scores_zscore_nofreqmult_2way_AUDSMPROD.npz")
TOP_SHUF = os.path.join(REPO, "decomposition",
                        "shuffle_scores_zscore_nofreqmult_2way_AUDSMPROD.npz")

BOT_TRUE = os.path.join(REPO, "decoding",
                        "true_scores_freqmult_zscore_weighted_2way2.npz")
BOT_SHUF = os.path.join(REPO, "decoding",
                        "shuffle_scores_freqmult_zscore_weighted_2way2.npz")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES = 2
BASELINE = 1 / N_CLASSES   # 0.5
YLIMS = (BASELINE - 0.2, BASELINE + 0.5)   # (0.3, 1.0)

# Six pairwise conditions (3 auditory-aligned + 3 go-aligned)
CONDS = [
    ["aud_ls", "aud_lm"], ["aud_ls", "aud_jl"], ["aud_lm", "aud_jl"],
    ["go_ls", "go_lm"],   ["go_ls", "go_jl"],   ["go_lm", "go_jl"],
]
COND_TITLES = {
    "aud_ls-aud_lm": "LS vs LM",
    "aud_ls-aud_jl": "LS vs JL",
    "aud_lm-aud_jl": "LM vs JL",
    "go_ls-go_lm":   "LS vs LM",
    "go_ls-go_jl":   "LS vs JL",
    "go_lm-go_jl":   "LM vs JL",
}

# Top row: electrode-group decoding
TOP_SERIES = [
    ("Production",    [0, 0, 1]),
    ("Sensory-Motor", [1, 0, 0]),
    ("Auditory",      [0, 1, 0]),
]

# Bottom row: component-weighted decoding
BOT_SERIES = [
    ("Auditory", "orange"),
    ("WM",       "k"),
    ("Motor",    "c"),
    ("Visual",   "y"),
]

# Styling (matches figure_2 / figure_4 / figure_5)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _acc(cm_arr: np.ndarray) -> np.ndarray:
    """Extract mean diagonal accuracy from (..., C, C) confusion matrices."""
    eye = np.eye(N_CLASSES, dtype=bool)
    return np.mean(cm_arr.T[eye].T, axis=2)


def _cond_key(name: str, cond) -> str:
    if isinstance(cond, list):
        return f"{name}-{'-'.join(cond)}"
    return f"{name}-{cond}"


def _times_true(cond_str: str) -> tuple[float, float]:
    """Time range for true-score traces (matches plot_all_scores)."""
    return (-0.4, 1.4)


def _times_shuffle(cond_str: str) -> tuple[float, float]:
    """Time range for shuffle overlay."""
    return (-0.5, 1.5)


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
gs = gridspec.GridSpec(
    2, 6, figure=fig, **GS_KWARGS,
    width_ratios=[1, 1, 1, 1, 1, 1],
)

# Epoch group labels (auditory and go) spanning 3 columns each
fig.text(0.28, 0.97, "Stimulus Onset", ha="center", fontsize=LABEL_SIZE,
         fontweight="bold")
fig.text(0.74, 0.97, "Go Cue", ha="center", fontsize=LABEL_SIZE,
         fontweight="bold")


def _plot_row(row_idx: int, true_scores: dict, shuf_scores: dict,
              series: list[tuple], row_tag: str,
              stat_func: callable) -> list[plt.Axes]:
    """Populate one row (6 panels) of pairwise decoding results."""
    axes = []
    for j, cond in enumerate(CONDS):
        ax = fig.add_subplot(gs[row_idx, j])
        axes.append(ax)

        cond_str = "-".join(cond)
        times_t = _times_true(cond_str)

        # ---- True-score traces ----
        for name, color in series:
            key = _cond_key(name, cond)
            if key not in true_scores:
                continue
            acc = _acc(true_scores[key])
            pl = acc.reshape(acc.shape[0], -1).T
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

            true_acc = _acc(true_scores[key])
            shuf_acc = _acc(shuf_scores[key])

            sig = time_perm_cluster(
                true_acc.mean(axis=1, keepdims=True).T,
                shuf_acc.T, 0.05, n_perm=10000,
                stat_func=stat_func,
            )[0]
            bars.append(sig)
            bar_colors.append(color)
            bar_times.append(times_t)

            window = np.lib.stride_tricks.sliding_window_view(
                shuf_acc, 20, axis=0)
            shuf_smooth = np.mean(window, axis=-1)
            times_sh = _times_shuffle(cond_str)
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

        # Column titles (top row only)
        if row_idx == 0:
            ax.set_title(COND_TITLES[cond_str], fontsize=LABEL_SIZE)
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            if "aud" in cond_str:
                ax.set_xlabel("Time from\nstimulus (s)", fontsize=LABEL_SIZE)
            else:
                ax.set_xlabel("Time from\ngo cue (s)", fontsize=LABEL_SIZE)

        # Y-axis label on leftmost column only
        if j == 0:
            ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
            ax.legend(fontsize=TICK_SIZE, loc="upper left",
                      framealpha=0.6)
        else:
            ax.set_yticklabels([])

    return axes


# ---------------------------------------------------------------------------
# Top row — electrode-group decoding
# ---------------------------------------------------------------------------
_stat_top = lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)
_plot_row(0, top_true, top_shuf, TOP_SERIES, "top", _stat_top)

# ---------------------------------------------------------------------------
# Bottom row — component-weighted decoding
# ---------------------------------------------------------------------------
_stat_bot = lambda x, y, axis: np.mean(x, axis=axis)
_plot_row(1, bot_true, bot_shuf, BOT_SERIES, "bottom", _stat_bot)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_6.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_6.png"), bbox_inches="tight", dpi=300)
plt.show()
