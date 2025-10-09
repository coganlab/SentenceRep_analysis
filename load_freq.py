import os
import numpy as np
import mne
from itertools import combinations

from ieeg.io import get_data
from analysis.grouping import group_elecs
from ieeg.arrays.label import Labels
from analysis.load import load_data
from analysis.decoding.utils import get_scores

# Add missing imports for decoding and plotting
from ieeg.decoding.decode import Decoder, plot_all_scores
from ieeg.decoding.models import PcaLdaClassification
from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.ensemble import plot_dist_bound
from analysis.utils.plotting import plot_horizontal_bars

exclude = [
    "D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
    "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
    "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
    "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
    "D0027-LPIF10", "D0026-RPG20", "D0026-RPG21", "D0026-RPG28",
    "D0026-RPG29", "D0026-RPG36", "D0007-RFG44"
]


fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
subjects = layout.get_subjects()
conds_all = {
    "resp": (-1, 1), "aud_ls": (-0.5, 1.5),
    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
    "go_jl": (-0.5, 1.5)
}
folder = 'stats_freq_hilbert'


def average_tfr_channels(tfr: mne.time_frequency.tfr.AverageTFR):
    info = mne.create_info(ch_names=['Average'], sfreq=tfr.info['sfreq'])
    avgd_data = np.nanmean(tfr.data, axis=0, keepdims=True)
    return mne.time_frequency.AverageTFRArray(info, avgd_data, tfr.times, tfr.freqs)

def name_from_idx(idx: list[int], chs: Labels):
    return [f"D{int(p[0][1:])}-{p[1]}" for p in
             (s.split("-") for s in chs[list(idx)])]

conds_all = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}

sigs = load_data(layout,folder, "mask", conds_all, 2,
                     out_type=bool, average=True, n_jobs=1)
zscores = load_data(layout, folder, "zscore", conds_all, 3,
                    out_type="float16", average=False, n_jobs=1)


AUD, SM, PROD, sig_chans, delay = group_elecs(sigs,
                                              [s for s in
                                               sigs.labels[1]
                                               if s not in exclude],
                                              sigs.labels[0])
idxs = {'SM': sorted(SM), 'AUD': sorted(AUD), 'PROD': sorted(PROD),
        'sig_chans': sorted(sig_chans), 'delay': sorted(delay)}

SM = sorted(SM)
AUD = sorted(AUD)
PROD = sorted(PROD)
sig_chans = sorted(sig_chans)

# %% plot data
do_plot=False
if do_plot:
    avg = np.nanmean(zscores, axis=(1, 4))
    times = np.linspace(-0.5, 1.5, 200)
    info = mne.create_info(ch_names=avg.labels[1].tolist(), sfreq=100)
    events = np.array([[0 + 200 * i, 0, i] for i in range(7)])
    event_id = dict(zip(avg.labels[0], range(7)))
    picked = mne.time_frequency.EpochsTFRArray(
        info, avg.__array__(), times, avg.labels[2],
        events=events, event_id=event_id, drop_log=tuple(() for _ in range(7)))
    x = picked['aud_ls'].pick(sorted(SM)).average()
    mne.time_frequency.AverageTFRArray(x.copy().pick(0).info,
                                       np.mean(x.data, axis=0, keepdims=True),
                                       x.times, x.freqs).plot(0)

# %% plot brain
plot_brain = False
if plot_brain:
    from ieeg.viz.mri import plot_on_average
    br = None
    for i, idx in enumerate([delay]):
        picks = name_from_idx(idx, zscores.labels[2])
        rgb = [1 if j == i else 0 for j in range(3)]
        br = plot_on_average(subjects, picks=picks, hemi='both', color=[rgb]*len(idx), fig=br)

# %% word decoding
decode = True
if decode:
    model = PcaLdaClassification(explained_variance=0.90, da_type='lda')
    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5,10, 2, 'train', model=model)
    # decoder = Decoder({'ls': 0, 'lm': 1, 'jl': 2},
    #                   5, 10, 1, 'train', model=model)
    n_classes = 4
    wh = -2
    baseline = 1 / n_classes
    true_scores = {}
    shuffle_score = {}
    names = ['Production','Sensory-Motor', 'Auditory']
    idxs = [PROD, SM, AUD]
    window_kwargs = {'window': 20,
                     'obs_axs': 2,
                     'normalize': 'true',
                     'n_jobs': 1,
                    'average_repetitions': False,
                     'step': 5}
    # conds = list(map(list, list(combinations(['aud_ls', 'aud_lm', 'aud_jl'], 2))
    #                   + list(
    #              combinations(['go_ls', 'go_lm', 'go_jl'], 2))))
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'],
             'resp']
    colors = [[0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5], [1, 0, 1]]
    suffix = '_zscore_nofreqmult_word_AUDSMPROD'
    true_name = 'true_scores' + suffix
    print('averaging...')
    data = np.nanmean(zscores, axis=-3, keepdims=True)
    print('done')
    if not os.path.exists(true_name + '.npz'):
        for values in get_scores(data
                                 , decoder, idxs, conds,
                                 names, on_gpu=True, shuffle=False,
                                 which=wh, **window_kwargs):
            key = decoder.current_job
            true_scores[key] = values
            np.savez(true_name, **true_scores)
    else:
        true_scores = dict(np.load(true_name + '.npz', allow_pickle=True))

    plots = {}
    for key, values in true_scores.items():
        if values is None:
            continue
        plots[key] = np.mean(values.T[np.eye(n_classes).astype(bool)].T, axis=2)
    fig, axs = plot_all_scores(plots, conds,
                               {n: i for n, i in zip(names, idxs)},
                               colors, "Word Decoding", ylims = (
            baseline-0.2, baseline + 0.6))

    # raise RuntimeError("Stop here to check true scores")
    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5, 100, 1, 'test', model=model)
    # decoder = Decoder({'ls': 0, 'lm': 1, 'jl': 2},
    #                   5, 50, 1, 'train', model=model)
    # suffix = '_freqmult_zscore_mult_conditions_2way_AUDSMPROD'
    shuffle_name = 'shuffle_scores' + suffix
    if not os.path.exists(shuffle_name + '.npz'):
        for values in get_scores(data, decoder, idxs, conds,
                                 names, which=wh, on_gpu=True,
                                 shuffle=True, **window_kwargs):
            # values = [values]
            # values = np.stack(values, axis=1)
            key = decoder.current_job
            shuffle_score[key] = values

        # shuffle_score['All-aud_ls-aud_lm'] = shuffle_score['Auditory-aud_ls-aud_lm']
        # shuffle_score['All-go_ls-go_lm'] = shuffle_score['Production-go_ls-go_lm']
        # shuffle_score['All-resp'] = shuffle_score['Production-resp']

        np.savez(shuffle_name, **shuffle_score)
    else:
        shuffle_score = dict(np.load(shuffle_name + '.npz', allow_pickle=True))
        # for k in list(shuffle_score.keys()):
        #     # if k.startswith('Auditory-'):
        #     #     shuffle_score['All-' + k[9:]] = shuffle_score[k].squeeze()
        #     shuffle_score[k] = shuffle_score[k].squeeze()


    # %% Time Sliding decoding significance

    signif = {}
    for cond, score in true_scores.items():
        true = np.mean(score.T[np.eye(n_classes).astype(bool)].T, axis=2)
        shuffle = np.mean(shuffle_score[cond].T[np.eye(n_classes).astype(bool)].T, axis=2)
        signif[cond] = time_perm_cluster(
            true.mean(axis=1, keepdims=True).T,
            shuffle.T, 0.05, n_perm=10000,
            stat_func=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)
        )[0]

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
            shuffle = np.mean(shuffle_score[name].T[np.eye(n_classes).astype(bool)].T, axis=2)
            # smooth the shuffle using a window
            window = np.lib.stride_tricks.sliding_window_view(shuffle, 20, 0)
            shuffle = np.mean(window, axis=-1)
            plot_dist_bound(shuffle, 'std', 'both', times, 0, ax=ax, color=colors[i], alpha=0.3)
            bars.append(signif[name])
        plot_horizontal_bars(ax, bars, 0.02, 'below')

    for ax in fig.axes:
        ax.axhline(1 / n_classes, color='k', linestyle='--')
