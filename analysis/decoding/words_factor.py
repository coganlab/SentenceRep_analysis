# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from analysis.grouping import GroupData
from analysis.data import dataloader
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, get_scores, plot_all_scores)
import torch
import matplotlib.pyplot as plt
import slicetca
from ieeg.viz.ensemble import plot_dist


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

    fpath = os.path.expanduser("~/Box/CoganLab")
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idx = sorted(list(sub.SM))
    aud_slice = slice(0, 175)
    colors = ['orange', 'y', 'k', 'c', 'm']
    colors_new = ['m', 'c', 'k', 'orange', 'y']
    conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
    scores = {c: None for c in colors}
    # neural_data_tensor, labels = dataloader(sub, idx, conds)
    # mask = ~torch.isnan(neural_data_tensor)
    # neural_data_tensor[torch.isnan(neural_data_tensor)] = 0

    n_components = (5,)
    best_seed = 123457
    window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                    'average_repetitions': False}

    # #
    # %% decompose the optimal model

    # losses, model = slicetca.decompose(neural_data_tensor,
    #                                    n_components,
    #                                    # (0, n_components[0], 0),
    #                                    seed=best_seed,
    #                                    positive=True,
    #                                    # min_std=5e-5,
    #                                    # iter_std=1000,
    #                                    learning_rate=1e-2,
    #                                    max_iter=10000,
    #                                    # batch_dim=0,
    #                                    batch_prop=0.33,
    #                                    batch_prop_decay=3,
    #                                    # weight_decay=1e-3,
    #                                    mask=mask,
    #                                    init_bias=0.01,
    #                                    initialization='uniform-positive',
    #                                    loss_function=torch.nn.HuberLoss(reduction='sum'),
    #                                    verbose=0
    #                                    )
    model = torch.load('model1.pt')

    T, W, H = model.get_components(numpy=True)[0]
    # %% plot the components
    cond_times = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5),
             'resp': (-1, 1)}
    timings = {'aud_ls': range(0, 200),
               'go_ls': range(400, 600),
               'resp': range(800, 1000)}
    fig, axs = plt.subplots(1, 3)

    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(cond_times.items()):
        ax = axs[j]
        for i in range(n_components[0]):
            fig = plot_dist(
                # H[i],
                model.construct_single_component(0, i).detach().cpu().numpy()[:, (W[i] / W.sum(0)) > 0.4][
                    ..., timings[cond]].reshape(-1, 200),
                ax=ax, color=colors[i], mode='sem', times=times)
        if j == 0:
            ax.legend()
            ax.set_ylabel("Z-Score (V)")
            ylims = ax.get_ylim()
        elif j == 1:
            ax.set_xlabel("Time(s)")
        ax.set_ylim(ylims)
        ax.set_title(cond)
    idxs = [torch.tensor(idx)[(W[i] / W.sum(0)) > 0.4].tolist() for i in range(5) ]

    # %% Time Sliding decoding for word tokens
    decode_conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
    scores2 = score2({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.9, 'lda', 5, 10, sub, idxs, decode_conds,
                                window_kwargs, scores, shuffle=False)
    dict_to_structured_array(scores, 'true_scores.npy')

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
    fig, axs = plot_all_scores(plots, decode_conds, {n: i for n, i in zip(colors, idxs)}, colors_new, "Word Decoding")

    for ax in fig.axes:
        ax.axhline(0.25, color='k', linestyle='--')
        # remove legend
        ax.legend().remove()

    # # %% Time Sliding decoding significance
    #
    # shuffle_score = np.load(data_dir + 'shuffle_score_short.npy', allow_pickle=True)[0]
    # shuffle_score = {name: shuffle_score[name] for name in shuffle_score.dtype.names}
    # signif = {}
    # for cond, score in scores.items():
    #     true = np.mean(score.T[np.eye(4).astype(bool)].T, axis=2)
    #     shuffle = np.mean(shuffle_score[cond].T[np.eye(4).astype(bool)].T, axis=2)
    #     signif[cond] = time_perm_cluster(true.T, shuffle.T, 0.001, stat_func=lambda x, y, axis: np.mean(x, axis=axis))
    #
    #
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
