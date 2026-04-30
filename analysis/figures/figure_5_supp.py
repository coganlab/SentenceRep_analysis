"""Figure 5 Supplement: Electrode-Group Word Decoding.

Layout (1 x 3):
  Electrode-group word decoding (Production / Sensory-Motor / Auditory) for
  three alignment conditions (aud, go, resp).
  Data source: load_freq.py pre-computed scores.
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist, plot_dist_bound
from analysis.utils.plotting import plot_horizontal_bars
from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE, DECOMPOSITION_DIR, DPI,
    XLABEL_STIMULUS, XLABEL_GO, XLABEL_RESPONSE,
)

# ---------------------------------------------------------------------------
# Paths to pre-computed score files
# ---------------------------------------------------------------------------
TOP_TRUE = os.path.join(DECOMPOSITION_DIR,
                        "true_scores_zscore_nofreqmult_word_AUDSMPROD4.npz")
TOP_SHUF = os.path.join(DECOMPOSITION_DIR,
                        "shuffle_scores_zscore_nofreqmult_word_AUDSMPROD4.npz")

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

TOP_SERIES = [
    ("Production",    [0, 0, 1]),
    ("Sensory-Motor", [1, 0, 0]),
    ("Auditory",      [0, 1, 0]),
]


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


def _times_shuffle(cond_str: str) -> tuple[float, float]:
    if cond_str.endswith("resp"):
        return (-1, 1)
    return (-0.5, 1.5)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
top_true = dict(np.load(TOP_TRUE, allow_pickle=True))
top_shuf = dict(np.load(TOP_SHUF, allow_pickle=True))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(18 * cm, 7.5 * cm))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25)

_stat = lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)

for j, cond in enumerate(CONDS):
    ax = fig.add_subplot(gs[0, j])
    cond_str = "-".join(cond) if isinstance(cond, list) else cond
    times_t = _times_true(cond_str)

    # ---- True-score traces ----
    for name, color in TOP_SERIES:
        key = _cond_key(name, cond)
        if key not in top_true:
            continue
        acc = _acc(top_true[key])
        pl = acc.reshape(acc.shape[0], -1).T
        plot_dist(pl, mode="std", times=times_t,
                  color=color, label=name, ax=ax)

    # ---- Shuffle overlay + significance bars ----
    bars = []
    for name, color in TOP_SERIES:
        key = _cond_key(name, cond)
        if key not in top_shuf:
            continue
        true_acc = _acc(top_true[key])
        shuf_acc = _acc(top_shuf[key])
        sig = time_perm_cluster(
            true_acc.mean(axis=1, keepdims=True).T,
            shuf_acc.T, 0.05, n_perm=10000,
            stat_func=_stat,
        )[0]
        bars.append(sig)
        window = np.lib.stride_tricks.sliding_window_view(
            shuf_acc, 20, axis=0)
        shuf_smooth = np.mean(window, axis=-1)
        plot_dist_bound(shuf_smooth, "std", "both",
                        _times_shuffle(cond_str), 0,
                        ax=ax, color='grey', alpha=0.2, linewidth=0)

    if bars:
        plot_horizontal_bars(ax, bars, 0.02, "below")

    ax.axhline(BASELINE, color="k", linestyle="--", linewidth=0.5)
    ax.set_ylim(*YLIMS)
    if cond_str == "resp":
        ax.set_xlabel(XLABEL_RESPONSE, fontsize=LABEL_SIZE)
    elif "aud" in cond_str:
        ax.set_xlabel(XLABEL_STIMULUS, fontsize=LABEL_SIZE)
    else:
        ax.set_xlabel(XLABEL_GO, fontsize=LABEL_SIZE)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x * 100:.0f}"))
    if j == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper left",
                  framealpha=0.6)
    else:
        ax.set_yticklabels([])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_5_supp.svg"), bbox_inches="tight", dpi=DPI)
fig.savefig(os.path.join(out_dir, "figure_5_supp.png"), bbox_inches="tight",
            dpi=DPI)
plt.show()
