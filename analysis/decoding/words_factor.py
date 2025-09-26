# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from analysis.grouping import group_elecs
from ieeg.decoding.decode import (Decoder, plot_all_scores)
from ieeg.decoding.models import PcaLdaClassification, PcaEstimateDecoder
from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist_bound, plot_dist
import matplotlib.pyplot as plt
from analysis.utils.plotting import plot_horizontal_bars
from analysis.decoding.utils import get_scores
import torch
import slicetca
from functools import partial
from itertools import combinations
from ieeg.io import get_data
from analysis.load import load_data, load_spec
from sklearn import set_config

set_config(enable_metadata_routing=True)


def weighted_preserve_stats(data, weights, axis=None):
    """
    Multiplies data along the specified axis by weights, then rescales
    to preserve the original mean and variance.

    Parameters:
        data (np.ndarray): The input data array.
        weights (np.ndarray): The weight vector.
        axis (int): The axis along which to multiply.

    Returns:
        np.ndarray: The weighted and rescaled data.
    """
    where = ~np.isnan(data)
    kwargs = {'where': where, 'dtype': 'f4'}
    w = weights / np.mean(weights)

    # Multiply along the specified axis
    if axis is None:
        data *= w
    else:
        data *= w.reshape([1 if i != axis else -1 for i in range(data.ndim)])

    # # Rescale to preserve mean and variance
    # weighted_mean = np.mean(data, **kwargs)
    # weighted_std = np.std(data, **kwargs)
    # data -= weighted_mean
    # data *= orig_std / weighted_std
    # data += orig_mean


if __name__ == '__main__':


    exclude = ["D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
               "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
               "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
               "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
               "D0027-LPIF10"]

    HOME = os.path.expanduser("~")
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        LAB_root = os.path.join(HOME, "workspace", "CoganLab")
        n = int(os.environ['SLURM_ARRAY_TASK_ID'])
        print(n)
    else:  # if not then set box directory
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
        n = 1

    log_dir = os.path.join(os.path.dirname(LAB_root), 'logs', str(n))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    layout = get_data('SentenceRep', root=LAB_root)

    conds_all = {"aud_ls": (-0.5, 1.5),
                     "aud_lm": (-0.5, 1.5),
                        "aud_jl": (-0.5, 1.5),
                     "go_ls": (-0.5, 1.5),
                 "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5),
                 }

    group = 'SM'
    folder = 'stats_freq_hilbert'

    sigs = load_data(layout, folder, "mask")
    AUD, SM, PROD, sig_chans, delay = group_elecs(sigs,
                                                  [s for s in
                                                   sigs.labels[1]
                                                   if s not in exclude],
                                                  sigs.labels[0])
    # idxs = {'SM': sorted(SM), 'AUD': sorted(AUD), 'PROD': sorted(PROD),
    #         'sig_chans': sorted(sig_chans), 'delay': sorted(delay)}
    # idx = idxs[group]
    zscores = load_data(layout, folder, "zscore")

    n_components = (4,)
    best_seed = 123457
    window_kwargs = {'window': 16, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': 1,
                    'average_repetitions': False, 'step': 8}

    # #
    # %% decompose the optimal model

    import pickle
    filename = f"{group}_chns.pkl"
    if os.path.exists(filename):
        labels = pickle.load(open(filename, 'rb'))
    else:
        labels = load_spec(group, list(conds_all.keys()), layout, folder=folder,
                           min_nan=1, n_jobs=-2)[-2]
        pickle.dump(labels, open(filename, 'wb'))
    state = torch.load('model_SM_freq.pt')
    shape = state['vectors.0.0'].shape[1:] + state['vectors.0.1'].shape[1:]
    n_components = (state['vectors.0.0'].shape[0],)
    model = slicetca.core.SliceTCA(
        dimensions=shape,
        ranks=(n_components[0], 0, 0, 0),
        positive=True,
        initialization='uniform-positive',
        dtype=torch.float32,
        lr=5e-4,
        weight_decay=partial(torch.optim.Adam, eps=1e-9),
        loss=torch.nn.L1Loss(reduction='mean'),
        init_bias=0.1,
        threshold=None,
        patience=None
    )
    model.load_state_dict(state)
    model[0]
    # model = torch.load('model_sig_chans.pt')

    W, H = model.get_components(numpy=True)[0]
    names = ['Auditory', 'WM', 'Motor', 'Visual']
    names = ['Auditory', 'Visual', 'WM', 'Motor']
    # fig, ax = plt.subplots(1,1)
    # for i in range(n_components[0]):
    #     subset = np.nonzero(W[i] / W.mean() > 0.05)[0]
    #     # subset = np.nonzero(W[i]/W.sum(0) > 0.5)[0]
    #     # subset = np.nonzero(W[i] == np.max(W, 0))[0]
    #     in_data = zscores[:, :, [labels[0][s] for s in subset]]
    #     print(f"{names[i]} component, {len(subset)} channels")
    #     weights = model.construct_single_component(0, i).detach().numpy()[
    #         subset]
    #     weights_aud = np.nanmean(weights[None, ..., None, :200], axis=2)
    #     weights_go = np.nanmean(weights[None, ..., None, 200:], axis=2)
    #     weighted_preserve_stats(in_data['aud_ls'], weights_aud)
    #     plot_dist(np.nanmean(in_data['aud_ls'].__array__(), axis=(0, 2, 3)), ax=ax)
    # raise RuntimeError("stop")

    # %% Time Sliding decoding for word tokens
    # decode_model = PcaLdaClassification(explained_variance=0.80, da_type='lda')
    decode_model = PcaEstimateDecoder(0.80, clf_params={'max_iter': 10000})
    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5, 4, 1, 'train', model=decode_model)
    true_scores = {}
    shuffle_scores = {}
    colors = ['orange', 'k', 'c', 'y', 'm', 'deeppink',
              'darkorange', 'lime', 'blue', 'red', 'purple'][:n_components[0]]

    real_names = ['Auditory', 'Visual', 'WM', 'Motor']
    assert W.shape[1] == len(labels[0])
    idx = [zscores.find(c, 2) for c in
           labels[0]]
    idxs = {c: idx for c in colors}
    window_kwargs = {'window': 20, 'obs_axs': 2, 'normalize': 'true',
                     'n_jobs': 10, 'oversample': True,
                     'average_repetitions': False, 'step': 5}
    # conds = list(map(list, list(combinations(['aud_ls', 'aud_lm', 'aud_jl'], 2))
    #                   + list(
    #              combinations(['go_ls', 'go_lm', 'go_jl'], 2))))
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm']]
    # colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]]
    # raise RuntimeError('stop')
    suffix ='_freqmult_zscore_weighted_words'
    n_classes = 4
    true_name = 'true_scores' + suffix

    baseline = 1 / n_classes

    if not os.path.exists(true_name + '.npz'):
        for i in range(n_components[0]):
            subset = np.nonzero(W[i] / W.mean() > 0.05)[0]
            # subset = np.nonzero(W[i]/W.sum(0) > 0.5)[0]
            # subset = np.nonzero(W[i] == np.max(W, 0))[0]
            in_data = zscores[:,:,[labels[0][s] for s in subset]]
            print(f"{names[i]} component, {len(subset)} channels")
            weights = model.construct_single_component(0, i).detach().numpy()[subset]
            weights_aud = np.nanmean(weights[None, ..., None, :200], axis=2)
            weights_go = np.nanmean(weights[None, ..., None, 200:], axis=2)
            weighted_preserve_stats(in_data['aud_ls'], weights_aud)
            weighted_preserve_stats(in_data['aud_lm'], weights_aud)

            weighted_preserve_stats(in_data['aud_jl'], weights_aud)
            weighted_preserve_stats(in_data['go_ls'], weights_go)
            weighted_preserve_stats(in_data['go_lm'], weights_go)
            weighted_preserve_stats(in_data['go_jl'], weights_go)
            # weighted_preserve_stats(in_data['resp'], W[i, subset], 1)
            # weighted_preserve_stats(in_data, weights, 2)
            for values in get_scores(in_data, decoder, [list(range(subset.sum()))], conds,
                                     [names[i]], on_gpu=False, shuffle=False,
                                     which=-2, **window_kwargs):
                key = decoder.current_job
                true_scores[key] = values
                np.savez(true_name, **true_scores)
    else:
        true_scores = dict(np.load(true_name + '.npz', allow_pickle=True))

    # # replace names with real names in true scores
    # temp = true_scores.copy()
    # true_scores = {}
    # for key, value in temp.items():
    #     for r, n in zip(real_names, names):
    #         if key.startswith(n):
    #             true_scores[key.replace(n, r)] = value

    plots = {}
    for key, values in true_scores.items():
        if values is None:
            continue
        plots[key] = np.mean(values.T[np.eye(n_classes).astype(bool)].T, axis=2)
    fig, axs = plot_all_scores(plots, conds,
                               {n: i for n, i in zip(names, idxs)},
                               colors, "Word Decoding", ylims=(
            baseline-0.2, baseline + 0.6))

    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5, 50, 1, 'train', model=decode_model)
    # raise RuntimeError("stop")
    shuffle_name = 'shuffle_scores' + suffix

    if not os.path.exists(shuffle_name + '.npz'):
        for i in range(n_components[0]):
            subset = np.nonzero(W[i]/W.sum(0) > 0.05)[0]
            # subset = np.nonzero(W[i] == np.max(W, 0))[0]

            in_data = zscores[:,:,[labels[0][s] for s in subset]]
            weights = model.construct_single_component(0, i).detach().numpy()[subset]
            weights_aud = np.nanmean(weights[None, ..., None, :200], axis=2)
            weights_go = np.nanmean(weights[None, ..., None, 200:], axis=2)
            weighted_preserve_stats(in_data['aud_ls'],
                                    weights_aud)
            weighted_preserve_stats(in_data['aud_lm'],
                                    weights_aud)
            weighted_preserve_stats(in_data['aud_jl'],
                                    weights_aud)
            weighted_preserve_stats(in_data['go_ls'],
                                    weights_go)
            weighted_preserve_stats(in_data['go_lm'],
                                    weights_go)
            weighted_preserve_stats(in_data['go_jl'],
                                    weights_go)
            for values in get_scores(in_data, decoder, [list(range(subset.sum()))], conds,
                                     [names[i]], on_gpu=True, shuffle=True,
                                     which=0, **window_kwargs):
                key = decoder.current_job
                shuffle_scores[key] = values

                np.savez(shuffle_name, **shuffle_scores)
    else:
        shuffle_scores = dict(np.load(shuffle_name + '.npz', allow_pickle=True))

        # shuffle_score['All-aud_ls-aud_lm'] = shuffle_score['Auditory-aud_ls-aud_lm']
        # shuffle_score['All-go_ls-go_lm'] = shuffle_score['Production-go_ls-go_lm']
        # shuffle_score['All-resp'] = shuffle_score['Production-resp']

    # Time Sliding decoding significance

    signif = {}
    pvals = {}
    for cond, score in true_scores.items():
        true = np.mean(score.T[np.eye(n_classes).astype(bool)].T, axis=2)
        shuffle = np.mean(shuffle_scores[cond].T[np.eye(n_classes).astype(bool)].T,
                          axis=2)
        signif[cond], pvals[cond] = time_perm_cluster(
            true.mean(axis=1, keepdims=True).T,
            shuffle.T, 0.05, n_perm=50000,
            stat_func=lambda x, y, axis: np.mean(x, axis=axis)
        )

    # Plot significance
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
            shuffle = np.mean(shuffle_scores[name].T[np.eye(n_classes).astype(bool)].T,
                              axis=2)
            # smooth the shuffle using a window
            window = np.lib.stride_tricks.sliding_window_view(shuffle, 20, 0)
            shuffle = np.mean(window, axis=-1)
            plot_dist_bound(shuffle, 'std', 'both', times, 0, ax=ax,
                            color=colors[i], alpha=0.3)
            bars.append(signif[name])
        plot_horizontal_bars(ax, bars, 0.02, 'below')

    for ax in fig.axes:
        ax.axhline(1 / n_classes, color='k', linestyle='--')
