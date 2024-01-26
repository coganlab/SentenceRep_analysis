# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os
from itertools import chain

import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.viz.utils import plot_dist
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, extract, concatenate_conditions, classes_from_labels, decode_and_score,
                               flatten_list, get_scores, plot_all_scores)


# %% Imports
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')
# sub['power'].array = scale(sub['power'].array, np.max(sub['zscore'].array), np.min(sub['zscore'].array))
true_scores = {}
all_data = []
colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
idxs = [list(idx & sub.grey_matter) for idx in idxs]
names = list(scores.keys())
decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', n_splits=5, n_repeats=10, oversample=True)
scorer = 'acc'
window_kwargs = {'window_size': 20, 'axis': -1, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                 'average_repetitions': False}

# %% Time Sliding decoding for word tokens
conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
fig, axs = plt.subplots(1, len(conds))
fig2, axs2 = plt.subplots(1, len(idxs))
if len(conds) == 1:
    axs = [axs]
    axs2 = [axs2] * len(idxs)
for i, (idx, ax2) in enumerate(zip(idxs, axs2)):
    all_conds = flatten_list(conds)
    x_data = extract(sub, all_conds, idx, decoder.n_splits, 'zscore', False)
    ax2.set_title(names[i])
    for cond, ax in zip(conds, axs):
        if isinstance(cond, list):
            X = concatenate_conditions(x_data, cond)
            cond = "-".join(cond)
        else:
            X = x_data[:, cond]
        all_data.append(X)

        cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))

        # Decoding
        score = decode_and_score(decoder, X, labels, scorer, **window_kwargs)
        scores[names[i]] = np.mean(score.copy(), axis=1)
        if cond == 'resp':
            times = (-0.9, 0.9)
        else:
            times = (-0.4, 1.4)
        pl_sc = np.reshape(scores[names[i]], (scores[names[i]].shape[0], -1)).T
        plot_dist(pl_sc, times=times,
                    color=colors[i], label=list(scores.keys())[i], ax=ax)
        plot_dist(pl_sc, times=times, label=cond, ax=ax2)
        true_scores["-".join([names[i], cond])] = score.copy()

        if i == len(conds) - 1:
            ax.legend()
            ax.set_title(cond)
            ax.set_ylim(0.1, 0.8)
    if i == 0:
        ax2.legend()
    ax2.set_ylim(0.1, 0.8)

axs[0].set_xlabel("Time from stim (s)")
axs[1].set_xlabel("Time from go (s)")
axs[2].set_xlabel("Time from response (s)")
axs[0].set_ylabel("Accuracy (%)")
fig.suptitle("Word Decoding")

# %% Time Sliding decoding significance
decoder_shuff = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda',
                        n_splits=5, n_repeats=250, oversample=True)
shuffle_score = get_scores(sub, decoder_shuff, idxs, conds, shuffle=True, **window_kwargs)
signif = {}
for cond, score in true_scores.items():
    true = np.mean(score, axis=2)
    shuffle = np.mean(shuffle_score[cond].T[np.eye(len(decoder.categories)).astype(bool)].T, axis=2)
    signif[cond] = time_perm_cluster(true.T, shuffle.T, 0.05, stat_func=lambda x, y, axis: np.mean(x, axis=axis))

# %% Plot significance
for cond, ax in zip(conds, axs):
    bars = []
    if isinstance(cond, list):
        cond = "-".join(cond)
    for i, idx in enumerate(idxs):
        bars.append(signif["-".join([names[i], cond])])
    plot_horizontal_bars(ax, bars, 0.05, 'below')

# %% horizontal lines
for ax in chain(axs, axs2):
    ax.axhline(1 / len(set(labels)), color='k', linestyle='--')
