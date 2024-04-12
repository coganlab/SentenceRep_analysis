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
idx = list(sub.SM)
subjects = np.unique(sub.keys['channel'][idx].astype('U5'))
idxs = [[i for i in idx if subj in sub.array.labels[3][i]] for subj in
        subjects]
names = subjects
colors = mat_colors[:len(names)]
scores = {subj: None for subj in names}
conds = [['aud_ls', 'aud_lm'], ]
window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                 'average_repetitions': False}

# %% Time Sliding decoding for word tokens

score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 50, sub, idxs,
      conds,
      window_kwargs, '../../test_scores.npy', scores,
      shuffle=False)

# %% Plot the results
import matplotlib.pyplot as plt
scores = {key: value for key, value in scores.items() if value is not None}
for thresh in [0.29, 0.3, 0.31, 0.32, 0.33]:
    plots = {}; bad = {}; allscore = {}
    for key, values in scores.items():
        result = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
        if not np.any(thresh < np.mean(result[:30], axis=1)):
            plots[key] = result
        else:
            bad[key] = result
        allscore[key] = result

    pos = {n: i for n, i in zip(names, idxs) if any(n in p for p in plots.keys())}
    fig, axs = plot_all_scores(plots, conds, pos, colors, "Individual Decoding")

    bad.keys()
    plt.figure()
    all_dat = np.array(list(v[:30] for v in allscore.values()))
    good_dat = np.array(list(v[:30] for v in plots.values()))
    plt.hist(np.mean(all_dat, axis=2).flatten(), bins=100)
    # plt.axvline(np.mean(all_dat), color='blue')
    plt.hist(np.mean(good_dat, axis=2).flatten(), bins=100)
    # plt.axvline(np.mean(good_dat), color='orange')
    plt.title(f"Threshold: {thresh}")

    # %% Time Sliding decoding for word tokens
    idxs2 = [[i for i in idx if sub.array.labels[3][i][:5] not in [b[:5] for b in bad.keys()]]]
    scores2 = {'All': None}
    score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 30, sub, idxs2,
          conds,
          window_kwargs, '../../out_scores.npy', scores2,
          shuffle=False)

    scores2 = {key: value for key, value in scores2.items() if value is not None}
    result = {}
    for key, values in scores2.items():
        result[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)

    fig, axs = plot_all_scores(result, conds, dict(All=idx), colors, "Word Decoding")