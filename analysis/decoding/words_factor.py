# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, get_scores, plot_all_scores, plot_dist_bound)
import torch
import slicetca
from ieeg.calc.mat import LabeledArray


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


def score1(categories, test_size, method, n_splits, n_repeats, sub,
          conds, weights, window_kwargs, scores_dict, shuffle=False):
    decoder = Decoder(categories, test_size, method, n_splits=n_splits, n_repeats=n_repeats)
    # names = list(scores_dict.keys())
    while len(scores_dict) > 0:
        scores_dict.popitem()
    for key, values in get_scores(sub, decoder, [list(sub.SM)], conds, ['Sensory-Motor'], weights, shuffle=shuffle, **window_kwargs):
        print(key)
        scores_dict[key] = values
    return scores_dict


def score2(categories, test_size, method, n_splits, n_repeats, sub, idxs,
          conds, window_kwargs, scores_dict, shuffle=False):
    decoder = Decoder(categories, test_size, method, n_splits=n_splits, n_repeats=n_repeats)
    names = list(scores_dict.keys())
    while len(scores_dict) > 0:
        scores_dict.popitem()
    for key, values in get_scores(sub, decoder, idxs, conds, names, shuffle=shuffle, **window_kwargs):
        print(key)
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
    colors = ['b', 'r', 'g', 'y', 'k', 'c', 'm']
    scores = {'Sensory-Motor': None}
    scores2 = {'Sensory-Motor': None}
    idxs = [sub.SM]
    idxs = [list(idx) for idx in idxs]
    names = list(scores.keys())
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
    window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                    'average_repetitions': False}

    # %%
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    zscore = np.nanmean(sub['zscore'].array, axis=(-4, -2))
    aud_slice = slice(0, 175)
    trainz = np.hstack([zscore['aud_ls', :, aud_slice],
                        zscore['aud_lm',:, aud_slice],
                        zscore['go_ls'],
                        zscore['resp']])
    data = trainz[idxs[0]]
    neural_data_tensor = torch.from_numpy(data
                                          / np.nanstd(data)
                                          ).to(device)

    losses, model = slicetca.decompose(
        neural_data_tensor,
        [5],
        seed=123456,
        positive=True,
        min_std=10 ** -6,
        learning_rate=5 * 10 ** -4,
        max_iter=10 ** 4,
        batch_prop=0.2,
        batch_prop_decay=5,
        initialization='uniform-positive',
        init_bias=0.001,
        loss_function=torch.nn.HuberLoss(reduction='none'))

    W, H = model.get_components(numpy=True)[0]
    sm_labels = zscore[:, idxs[0]].labels[1]
    W = LabeledArray(W, [np.arange(W.shape[0]), sm_labels])

    idxs = [[(sub.array.labels[3] == s).nonzero()[0][0] for s in W.labels[1][np.argmax(W,0) == i]] for i in range(5) ]

    # %% Time Sliding decoding for word tokens

    scores = score1({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.9, 'lda', 5, 10, sub, conds,
                               W, window_kwargs, scores, shuffle=False)
    dict_to_structured_array(scores, 'true_scores.npy')

    scores2 = score2({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.9, 'lda', 5, 10, sub, idxs, conds,
                                window_kwargs, scores2, shuffle=False)

    # %% Plotting
    data_dir = ''
    # true_scores = np.load(data_dir + 'true_scores_short.npy', allow_pickle=True)[0]
    # true_scores = {name: true_scores[name] for name in true_scores.dtype.names}

    plots = {}
    for key, values in scores.items():
        if values is None:
            continue
        keys = key.split('-')
        keys.insert(2, keys.pop(-1))
        key = '-'.join(keys)
        plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
    names = [f"Sensory-Motor-{i}" for i in range(5)]
    fig, axs = plot_all_scores(plots, conds, {n: i for n, i in zip(names, idxs)}, colors[:5], "Word Decoding")

    for ax in fig.axes:
        ax.axhline(0.25, color='k', linestyle='--')

    # %% Time Sliding decoding significance

    shuffle_score = np.load(data_dir + 'shuffle_score_short.npy', allow_pickle=True)[0]
    shuffle_score = {name: shuffle_score[name] for name in shuffle_score.dtype.names}
    signif = {}
    for cond, score in scores.items():
        true = np.mean(score.T[np.eye(4).astype(bool)].T, axis=2)
        shuffle = np.mean(shuffle_score[cond].T[np.eye(4).astype(bool)].T, axis=2)
        signif[cond] = time_perm_cluster(true.T, shuffle.T, 0.001, stat_func=lambda x, y, axis: np.mean(x, axis=axis))

    # # %% Plot significance
    # for cond, ax in zip(conds, axs):
    #     bars = []
    #     if isinstance(cond, list):
    #         cond = "-".join(cond)
    #     for i, idx in enumerate(idxs):
    #         name = "-".join([names[i], cond])
    #         if name.endswith('resp'):
    #             times = (-1, 1)
    #         else:
    #             times = (-0.5, 1.5)
    #         shuffle = np.mean(shuffle_score[name].T[np.eye(4).astype(bool)].T, axis=2)
    #         # smooth the shuffle using a window
    #         window = np.lib.stride_tricks.sliding_window_view(shuffle, 20, 0)
    #         shuffle = np.mean(window, axis=-1)
    #         plot_dist_bound(shuffle, 'std', 'both', times, 0, ax=ax, color=colors[i], alpha=0.3)
    #         bars.append(signif[name])
    #     plot_horizontal_bars(ax, bars, 0.05, 'below')
