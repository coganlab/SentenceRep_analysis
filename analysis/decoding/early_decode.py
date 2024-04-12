from ieeg.viz.parula import mat_colors
import numpy as np
import os

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, get_scores, plot_all_scores, plot_dist_bound)
from analysis.decoding.words import dict_to_structured_array, score

# %% Imports
box = os.path.expanduser(os.path.join("~", "Box"))
fpath = os.path.join(box, "CoganLab")
subjects_dir = os.path.join(box, "ECoG_Recon")
sub = GroupData.from_intermediates(
    "SentenceRep", fpath, folder='stats', subjects_dir=subjects_dir)
all_data = []
idx = list(sub.SM & sub.grey_matter)
subjects = np.unique(sub.keys['channel'][idx].astype('U5'))
idxs = [[i for i in idx if subj not in sub.array.labels[3][i]] for subj in
        subjects]
names = subjects
colors = mat_colors[:len(names)]
scores = {subj: None for subj in names}
conds = [['aud_ls', 'aud_lm'], ]
window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                 'average_repetitions': False}

# %% Time Sliding decoding for word tokens

score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 10, sub, idxs,
      conds,
      window_kwargs, '../../test_scores.npy', scores,
      shuffle=False)

# %% Plot the results

plots = {}
for key, values in scores.items():
    if values is None:
        continue
    plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
fig, axs = plot_all_scores(plots, conds, {n: i for n, i in zip(names, idxs)}, colors, "Word Decoding")
