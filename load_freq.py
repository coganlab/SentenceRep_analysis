import os
import numpy as np
import cupy as cp
import mne

from sklearn import config_context
from ieeg.io import get_data, DataLoader
from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray, combine, Labels
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.decoding.decode import Decoder, flatten_list, extract, classes_from_labels, concatenate_conditions, plot_all_scores
from ieeg.viz.ensemble import plot_dist_bound
from ieeg.io import dict_to_structured_array

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
subjects = layout.get_subjects()
all_conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}

def load_data(datatype: str, out_type: type | str = float, average: bool = True):
    loader = DataLoader(layout, all_conds, datatype, average, 'stats_freq',
                       '.h5')
    zscore = loader.load_dict(dtype=out_type, n_jobs=12)
    if average:
        zscore_ave = combine(zscore, (0, 2))
        for key in zscore_ave.keys():
            for k in zscore_ave[key].keys():
                for f in zscore_ave[key][k].keys():
                    zscore_ave[key][k][f] = zscore_ave[key][k][f][..., :200]
    else:
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

# loader = DataLoader(layout, conds, "zscore", False, 'stats_freq',
#                     '.h5')
# zscore = loader.load_dict(dtype="float16")
# zscore2 = combine(zscore, (0, 3))
# zscore = LabeledArray.from_dict(zscore2, dtype="float16")
if not os.path.exists("zscores.npy"):
    zscores = load_data("zscore", "float16", False)
    zscores.tofile("zscores")
zscores = LabeledArray.fromfile("zscores", mmap_mode='r')

sigs = load_data("significance", out_type=bool)

AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])

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
    picked['aud_ls'].pick(AUD).average().plot(0)

# %% plot brain
plot_brain = False
if plot_brain:
    from ieeg.viz.mri import plot_on_average
    br = None
    for i, idx in enumerate([SM, AUD, PROD]):
        picks = name_from_idx(idx, avg.labels[1])
        rgb = [1 if j == i else 0 for j in range(3)]
        br = plot_on_average(subjects, picks=picks, hemi='both', color=[rgb]*len(idx), fig=br)

# %% word decoding
decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
                  5, 10, 1, 'train', explained_variance=0.8, da_type='lda')
scores_dict = {}
names = ['Production','Sensory-Motor', 'Auditory', 'All']
idxs = [PROD, SM, AUD, sig_chans]
window_kwargs = {'window': 20, 'obs_axs': 2, 'normalize': 'true', 'n_jobs': 1,
                    'average_repetitions': False, 'step': 5}
conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.5]]

for name, idx in zip(names, idxs):
    all_conds = flatten_list(conds)
    x_data = extract(zscores, all_conds, 4, idx, decoder.n_splits,
                     False)
    for cond in conds:
        if isinstance(cond, list):
            X = concatenate_conditions(x_data, cond, 0, 3)
            cond = "-".join(cond)
        else:
            X = x_data[cond,].dropna()
        cats, labels = classes_from_labels(X.labels[-2], crop=slice(0, 4))

        # Decoding
        # score = decoder.cv_cm(X.__array__(), labels, **window_kwargs)
        x_in = cp.asarray(X.__array__())
        y_in = cp.asarray(labels)
        with config_context(array_api_dispatch=True):
            score = decoder.cv_cm(x_in, y_in, **window_kwargs)
        key = "-".join([name, cond])
        scores_dict[key] = score.get()

dict_to_structured_array(scores_dict, 'true_scores_freq.npy')

plots = {}
for key, values in scores_dict.items():
    if values is None:
        continue
    plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
fig, axs = plot_all_scores(plots, conds,
                           {n: i for n, i in zip(names, idxs)},
                           colors, "Word Decoding")
