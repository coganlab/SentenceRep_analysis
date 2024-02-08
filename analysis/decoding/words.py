# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, get_scores, plot_all_scores)


def dict_to_structured_array(dict_matrices, filename='structured_array.npy'):
    # Get the keys and shapes
    keys = list(dict_matrices.keys())
    shape = dict_matrices[keys[0]].shape

    # Create a data type for the structured array
    dt = np.dtype([(key, dict_matrices[key].dtype, shape) for key in keys])

    # Create the structured array
    structured_array = np.zeros((1,), dtype=dt)

    # Fill the structured array
    for key in keys:
        structured_array[key] = dict_matrices[key]

    # Save the structured array to a file
    np.save(filename, structured_array)

# %% Imports
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
all_data = []
colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None , 'All': None}
idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
idxs = [list(idx & sub.grey_matter) for idx in idxs]
names = list(scores.keys())
conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                 'average_repetitions': False}

# %% Time Sliding decoding for word tokens

decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', n_splits=5, n_repeats=10)
true_scores = {}
plots = {}
scores = get_scores(sub, decoder, idxs, conds, **window_kwargs)
for cond, score in scores:
    print(cond)
    true_scores[cond] = score
    plots[cond] = np.mean(score.T[np.eye(len(decoder.categories)).astype(bool)].T, axis=2)
fig, axs = plot_all_scores(plots, conds, {n: i for n, i in zip(names, idxs)}, colors)
dict_to_structured_array(true_scores, 'true_scores.npy')

# %% Time Sliding decoding significance
decoder_shuff = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda',
                        n_splits=5, n_repeats=250)
shuffle_score = {}
scores = get_scores(sub, decoder_shuff, idxs, conds, shuffle=True, **window_kwargs)
for cond, score in scores:
    print(cond)
    shuffle_score[cond] = score
dict_to_structured_array(shuffle_score, 'shuffle_score.npy')

# %% Time Sliding decoding significance
signif = {}
for cond, score in true_scores.items():
    true = np.mean(score.T[np.eye(len(decoder.categories)).astype(bool)].T, axis=2)
    shuffle = np.mean(shuffle_score[cond].T[np.eye(len(decoder.categories)).astype(bool)].T, axis=2)
    signif[cond] = time_perm_cluster(true.T, shuffle.T, 0.001, stat_func=lambda x, y, axis: np.mean(x, axis=axis))

# %% Plot significance
for cond, ax in zip(conds, axs):
    bars = []
    if isinstance(cond, list):
        cond = "-".join(cond)
    for i, idx in enumerate(idxs):
        bars.append(signif["-".join([names[i], cond])])
    plot_horizontal_bars(ax, bars, 0.05, 'below')

# %% horizontal lines
for ax in axs:
    ax.axhline(1 / len(decoder.categories), color='k', linestyle='--')
