# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from ieeg.viz.utils import plot_dist
from ieeg.calc.mat import Labels
from . import (Decoder, decode_and_score, extract, concatenate_conditions,
               classes_from_labels)


# %% Imports
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')
# sub['power'].array = scale(sub['power'].array, np.max(sub['zscore'].array), np.min(sub['zscore'].array))
all_scores = {}
all_data = []
colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
idxs = [list(idx & sub.grey_matter) for idx in idxs]
names = list(scores.keys())
decoder = Decoder({'ls': 0, 'lm': 1, 'jl': 2}, 0.8, 'lda', n_splits=5, n_repeats=5, oversample=True)
scorer = 'acc'
window_kwargs = {'window_size': 20, 'axis': -1, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2}


# %% Time Sliding decoding for conditions
conds_aud = ['aud_ls', 'aud_lm']
conds_go = ['go_ls', 'go_lm']
fig, ax = plt.subplots(1, 2)
for i, idx in enumerate(idxs):

    # organize data
    all_data = extract(sub, conds_aud + conds_go, idx, 5, 'zscore', False)
    aud_data = concatenate_conditions(all_data, conds_aud, 1)
    aud_data.labels[1] = Labels([l.replace('aud_', '') for l in aud_data.labels[1]])
    go_data = concatenate_conditions(all_data, conds_go)
    go_data.labels[1] = Labels([l.replace('go_', '') for l in go_data.labels[1]])
    common = np.array([l for l in aud_data.labels[1] if l in go_data.labels[1]])
    x_data = aud_data[..., :175].concatenate(go_data[:, common], axis=2)
    cats, labels = classes_from_labels(x_data.labels[1], crop=slice(-2, None), which=1)
    # cats['ls'] = cats['lm'] # if you want to combine ls and lm
    decoder.categories = cats

    # Decoding
    score = decode_and_score(decoder, x_data, labels, scorer, **window_kwargs)
    scores[names[i]] = score.copy()
    pl_sc = np.reshape(scores[names[i]], (scores[names[i]].shape[0], -1)).T

    plot_dist(pl_sc[:, :165], times=(-0.4, 1.25), color=colors[i], label=list(scores.keys())[i], ax=ax[0])
    plot_dist(pl_sc[:, 166:], times=(-0.5, 1.4), color=colors[i],
              label=list(scores.keys())[i], ax=ax[1])
ax[0].set_xlabel("Time from stim (s)")
ax[1].set_xlabel("Time from go (s)")
ax[0].set_ylabel("Accuracy (%)")
fig.suptitle("Word Decoding")

# draw horizontal dotted lines at chance
ax[0].axhline(1/2, color='k', linestyle='--')
ax[1].axhline(1/2, color='k', linestyle='--')