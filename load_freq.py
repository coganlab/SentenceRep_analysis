import os
import numpy as np
import mne

from ieeg.io import get_data
from analysis.utils.mat_load import DataLoader
from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray, combine, Labels

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
subjects = layout.get_subjects()
conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}

def load_data(datatype: str, out_type: type | str = float, average: bool = True):
    loader = DataLoader(layout, conds, datatype, average, 'stats_freq',
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
avg = np.nanmean(zscores, axis=(1, 4))
sigs = load_data("significance", out_type=bool)

AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])

SM = sorted(SM)
AUD = sorted(AUD)
PROD = sorted(PROD)
sig_chans = sorted(sig_chans)

times = np.linspace(-0.5, 1.5, 200)

info = mne.create_info(ch_names=avg.labels[1].tolist(), sfreq=100)
events = np.array([[0 + 200 * i, 0, i] for i in range(7)])
event_id = dict(zip(avg.labels[0], range(7)))
picked = mne.time_frequency.EpochsTFRArray(
    info, avg.__array__(), times, avg.labels[2],
    events=events, event_id=event_id, drop_log=tuple(() for _ in range(7)))

picked['aud_ls'].pick(AUD).average().plot(0)
# chan_grid(picked['aud_ls'].average(), yscale='log', vlim=(0, 1), cmap=parula_map)
# all_spec = [picked['aud_ls'].pick_channels([pick]) for pick in picks]
# for sub in subjects:
#     picks = tuple(p for p in zscores.labels[1][AUD].tolist() if sub in p)
#     all_avg = average_tfr_channels(picked['aud_ls'].pick_channels(picks).average())
#     all_avg.plot('Average', cmap=parula_map, vlim=(0, 1), title=sub)
# all_avg = average_tfr_channels(picked['go_ls'].pick_channels(picks).average())
# all_avg.plot('Average', cmap=parula_map, vlim=(0, 1))
#
# picks_old = (p.split('-') for p in picks)
# picks = [f"D{int(p[0][1:])}-{p[1]}" for p in picks_old]
from ieeg.viz.mri import plot_on_average
br = None
for i, idx in enumerate([SM, AUD, PROD]):
    picks = name_from_idx(idx, avg.labels[1])
    rgb = [1 if j == i else 0 for j in range(3)]
    br = plot_on_average(subjects, picks=picks, hemi='both', color=[rgb]*len(idx), fig=br)
