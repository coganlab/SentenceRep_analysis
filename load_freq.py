import os
import numpy as np
import mne

from ieeg.io import get_data, DataLoader
from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray, combine, Labels
from ieeg.decoding.decode import Decoder, get_scores, plot_all_scores

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
subjects = layout.get_subjects()
all_conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}
folder = 'stats_freq_multitaper'

def load_data(datatype: str, out_type: type | str = float,
              average: bool = True, n_jobs: int = 12):
    conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}
    loader = DataLoader(layout, conds, datatype, average, folder,
                       '.h5')
    zscore = loader.load_dict(dtype=out_type, n_jobs=n_jobs)
    # if average:
    #     zscore_ave = combine(zscore, (0, 2))
    #     for key in zscore_ave.keys():
    #         for k in zscore_ave[key].keys():
    #             for f in zscore_ave[key][k].keys():
    #                 zscore_ave[key][k][f] = zscore_ave[key][k][f][..., :200]
    # else:
    zscore_ave = combine(zscore, (0, 3))

    del zscore
    return LabeledArray.from_dict(zscore_ave, dtype=out_type)

def average_tfr_channels(tfr: mne.time_frequency.tfr.AverageTFR):
    info = mne.create_info(ch_names=['Average'], sfreq=tfr.info['sfreq'])
    avgd_data = np.nanmean(tfr.data, axis=0, keepdims=True)
    return mne.time_frequency.AverageTFRArray(info, avgd_data, tfr.times, tfr.freqs)

def name_from_idx(idx: list[int], chs: Labels):
    return [f"D{int(p[0][1:])}-{p[1]}" for p in
             (s.split("-") for s in chs[idx])]

conds_all = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}
loader = DataLoader(layout, conds_all, 'significance', True, folder,
                   '.h5')
filemask = os.path.join(layout.root, 'derivatives', folder, 'combined', 'mask')
if not os.path.exists(filemask + ".npy"):
    sigs = LabeledArray.from_dict(combine(loader.load_dict(
        dtype=bool, n_jobs=-1), (0, 2)), dtype=bool)
    sigs.tofile(filemask)
else:
    sigs = LabeledArray.fromfile(filemask)

AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])
idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans}

filez = os.path.join(layout.root, 'derivatives', folder, 'combined', 'zscores')
if not os.path.exists(filez + ".npy"):
    zscores = load_data("zscore", "float16", False, -1)
    zscores.tofile(filez)
zscores = LabeledArray.fromfile(filez, mmap_mode='r')

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
    for i, idx in enumerate([SM, AUD, PROD]):
        picks = name_from_idx(idx, zscores.labels[2])
        rgb = [1 if j == i else 0 for j in range(3)]
        br = plot_on_average(subjects, picks=picks, hemi='both', color=[rgb]*len(idx), fig=br)

# %% word decoding
decode = False
if decode:
    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5, 10, 1, 'train', explained_variance=0.8, da_type='lda')
    scores_dict = {}
    scores_dict2 = {}
    names = ['Production','Sensory-Motor', 'Auditory', 'All']
    idxs = [PROD, SM, AUD, sig_chans]
    window_kwargs = {'window': 20, 'obs_axs': 2, 'normalize': 'true', 'n_jobs': 1,
                        'average_repetitions': False, 'step': 5}
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
    colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]]

    if not os.path.exists("true_scores_freq.npz"):
        for values in get_scores(zscores, decoder, idxs, conds,
                                 names, on_gpu=True, shuffle=False, **window_kwargs):
            key = decoder.current_job
            scores_dict[key] = values

        np.savez('true_scores_freq', **scores_dict)
    else:
        scores_dict = dict(np.load("true_scores_freq.npz", allow_pickle=True))

    plots = {}
    for key, values in scores_dict.items():
        if values is None:
            continue
        plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
    fig, axs = plot_all_scores(plots, conds,
                               {n: i for n, i in zip(names, idxs)},
                               colors, "Word Decoding")

    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5, 50, 1, 'train', explained_variance=0.8, da_type='lda')
    if not os.path.exists("shuffle_scores_freq.npz"):
        for values in get_scores(zscores, decoder, idxs, conds,
                                 names, on_gpu=True, shuffle=True, **window_kwargs):
            key = decoder.current_job
            scores_dict2[key] = values

        np.savez('shuffle_scores_freq', **scores_dict2)
    else:
        scores_dict2 = dict(np.load("shuffle_scores_freq.npy", allow_pickle=True))
