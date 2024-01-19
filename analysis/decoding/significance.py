# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from analysis.grouping import GroupData
from analysis.decoding import Decoder, get_scores


# %% Imports
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')
# sub['power'].array = scale(sub['power'].array, np.max(sub['zscore'].array), np.min(sub['zscore'].array))
colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
idxs = [sub.AUD, sub.SM, sub.PROD]  # , sub.sig_chans]
idxs = [list(idx & sub.grey_matter) for idx in idxs]
scorer = 'acc'
window_kwargs = {'window_size': 20, 'axis': -1, 'obs_axs': 1, 'n_jobs': -2,
                 'normalize': 'true', 'average_repetitions': False}

# %% Time Sliding decoding for word tokens
conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda',
                  n_splits=5, n_repeats=1, oversample=True)
true_score = get_scores(sub, decoder, idxs, conds, shuffle=False, **window_kwargs)

decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda',
                  n_splits=5, n_repeats=250, oversample=True)
shuffle_score = get_scores(sub, decoder, idxs, conds, shuffle=True, **window_kwargs)

# %% Time perm cluster stats
cond = 'Sensory-Motor-resp'
true = np.mean(true_score[cond][..., np.eye(len(decoder.categories)).astype(bool)], axis=2)
shuffle = np.mean(shuffle_score[cond][..., np.eye(len(decoder.categories)).astype(bool)], axis=2)
signif = np.mean(true < shuffle, axis=1) < 0.05
