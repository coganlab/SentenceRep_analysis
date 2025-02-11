# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.calc.stats import time_perm_cluster
from ieeg.decoding.decode import Decoder, plot_all_scores, get_scores
from ieeg.viz.ensemble import plot_dist_bound


def score(categories, test_size, method, n_splits, n_repeats, sub, idxs,
          conds, window_kwargs, scores_dict, gpu=False, shuffle=False, **extra_kwargs):
    decoder = Decoder(categories, n_splits=n_splits, n_repeats=n_repeats,
                      explained_variance=test_size, da_type=method, **extra_kwargs)
    names = list(scores_dict.keys())
    while len(scores_dict) > 0:
        scores_dict.popitem()
    for values in get_scores(sub.array['zscore'], decoder, idxs, conds,
                                  names, on_gpu=gpu, shuffle=shuffle, **window_kwargs):
        scores_dict[key] = values
    return scores_dict


if __name__ == '__main__':

    # %% Imports
    box = os.path.expanduser(os.path.join("~","Box"))
    fpath = os.path.join(box, "CoganLab")
    subjects_dir = os.path.join(box, "ECoG_Recon")
    sub = GroupData.from_intermediates(
        "SentenceRep", fpath, folder='stats', subjects_dir=subjects_dir)
    all_data = []
    colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
    scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
    scores2 = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
    idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
    idxs = [list(idx) for idx in idxs]
    names = list(scores.keys())
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
    window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': 1,
                    'average_repetitions': False}

    # %% Time Sliding decoding for word tokens

    scores = score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 10, sub, idxs, conds,
                                window_kwargs, scores,
                                shuffle=False, gpu=True)
    np.savez('true_scores.npz', **scores)
    scores2 = score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
                                    0.8, 'lda', 5, 50, sub, idxs, conds,
                                    window_kwargs, scores2,
                                    shuffle=True, gpu=True)
    np.savez('shuffle_score.npz', **scores2)

    # %% Plotting
    data_dir = ''
    true_scores = dict(np.load(data_dir + 'true_scores.npz', allow_pickle=True))

    plots = {}
    for key, values in true_scores.items():
        if values is None:
            continue
        plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
    fig, axs = plot_all_scores(plots, conds, {n: i for n, i in zip(names, idxs)}, colors, "Word Decoding")


    # %% Time Sliding decoding significance

    shuffle_score = dict(np.load(data_dir + 'shuffle_score.npz.', allow_pickle=True))
    signif = {}
    for cond, score in true_scores.items():
        true = np.mean(score.T[np.eye(4).astype(bool)].T, axis=2)
        shuffle = np.mean(shuffle_score[cond].T[np.eye(4).astype(bool)].T, axis=2)
        signif[cond] = time_perm_cluster(
            true.T, shuffle.T, 0.01,
            stat_func=lambda x, y, axis: np.mean(x, axis=axis))[0]

    # %% Plot significance
    for cond, ax in zip(conds, axs):
        bars = []
        if isinstance(cond, list):
            cond = "-".join(cond)
        for i, idx in enumerate(idxs):
            name = "-".join([names[i], cond])
            if name.endswith('resp'):
                times = (-1, 1)
            else:
                times = (-0.5, 1.5)
            shuffle = np.mean(shuffle_score[name].T[np.eye(4).astype(bool)].T, axis=2)
            # smooth the shuffle using a window
            window = np.lib.stride_tricks.sliding_window_view(shuffle, 20, 0)
            shuffle = np.mean(window, axis=-1)
            plot_dist_bound(shuffle, 'std', 'both', times, 0, ax=ax, color=colors[i], alpha=0.3)
            bars.append(signif[name])
        plot_horizontal_bars(ax, bars, 0.05, 'below')

    for ax in fig.axes:
        ax.axhline(0.25, color='k', linestyle='--')
