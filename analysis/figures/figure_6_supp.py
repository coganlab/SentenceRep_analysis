"""Figure 6 Supplement: Electrode-Group Pairwise Condition Decoding.

Layout (1 x 6):
  Electrode-group 2-way condition decoding (Production / Sensory-Motor /
  Auditory) for six pairwise comparisons.
  Data source: load_freq.py pre-computed scores.

Columns (6 pairwise condition comparisons):
  aud: LS vs LM, LS vs JL, LM vs JL
  go : LS vs LM, LS vs JL, LM vs JL
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist, plot_dist_bound
from analysis.utils.plotting import plot_horizontal_bars
from analysis.figures.config import DECOMPOSITION_DIR

# ---------------------------------------------------------------------------
# Paths to pre-computed score files
# ---------------------------------------------------------------------------
TOP_TRUE = os.path.join(DECOMPOSITION_DIR,
                        "true_scores_zscore_nofreqmult_2way_AUDSMPROD.npz")
TOP_SHUF = os.path.join(DECOMPOSITION_DIR,
                        "shuffle_scores_zscore_nofreqmult_2way_AUDSMPROD.npz")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES = 2
BASELINE = 1 / N_CLASSES
YLIMS = (BASELINE - 0.2, BASELINE + 0.5)

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

TOP_SERIES = [
    ("Production",    [0, 0, 1]),
    ("Sensory-Motor", [1, 0, 0]),
    ("Auditory",      [0, 1, 0]),
]

# Styling (matches figure_2 / figure_4 / figure_5)
cm = 1 / 2.54
LABEL_SIZE = 7
TICK_SIZE = 5


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
top_true = dict(np.load(TOP_TRUE, allow_pickle=True))
top_shuf = dict(np.load(TOP_SHUF, allow_pickle=True))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18 * cm, 6 * cm))
gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.15)

fig.text(0.28, 0.97, "Stimulus Onset", ha="center", fontsize=LABEL_SIZE,
         fontweight="bold")
fig.text(0.74, 0.97, "Go Cue", ha="center", fontsize=LABEL_SIZE,
         fontweight="bold")

_stat = lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)

for j, cond in enumerate(CONDS):
    ax = fig.add_subplot(gs[0, j])
    cond_str = "-".join(cond)
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
                        ax=ax, color=color, alpha=0.3)

    if bars:
        plot_horizontal_bars(ax, bars, 0.02, "below")

    ax.axhline(BASELINE, color="k", linestyle="--", linewidth=0.5)
    ax.set_ylim(*YLIMS)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title(COND_TITLES[cond_str], fontsize=LABEL_SIZE)

    if "aud" in cond_str:
        ax.set_xlabel("Time from\nstimulus (s)", fontsize=LABEL_SIZE)
    else:
        ax.set_xlabel("Time from\ngo cue (s)", fontsize=LABEL_SIZE)

    if j == 0:
        ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE - 2, loc="upper left",
                  framealpha=0.6)
    else:
        ax.set_yticklabels([])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "figure_6_supp.svg"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure_6_supp.png"), bbox_inches="tight",
            dpi=300)
plt.show()
