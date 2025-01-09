import os
import numpy as np
import scipy.stats as st
import mne

from ieeg.io import get_data
from analysis.utils.mat_load import DataLoader
from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray, combine

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
subjects = layout.get_subjects()
conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}

def load_data(datatype: str, out_type = float):
    loader = DataLoader(layout, conds, datatype, True, 'stats_freq',
                       '.h5')
    zscore = loader.load_dict()
    zscore_ave = combine(zscore, (0, 2))
    for key in zscore_ave.keys():
        for k in zscore_ave[key].keys():
            for f in zscore_ave[key][k].keys():
                zscore_ave[key][k][f] = zscore_ave[key][k][f][:200]
    del zscore
    return LabeledArray.from_dict(zscore_ave, dtype=out_type)

def average_tfr_channels(tfr: mne.time_frequency.tfr.AverageTFR):
    info = mne.create_info(ch_names=['Average'], sfreq=tfr.info['sfreq'])
    avgd_data = np.nanmean(tfr.data, axis=0, keepdims=True)
    return mne.time_frequency.AverageTFRArray(info, avgd_data, tfr.times, tfr.freqs)

sigs = load_data("significance", out_type=bool)

AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])

SM = sorted(SM)
AUD = sorted(AUD)
PROD = sorted(PROD)
sig_chans = sorted(sig_chans)

pvals = load_data("pval", out_type=float)
data = np.where(pvals > 0.9999, 0.9999, pvals)
zscores = LabeledArray(st.norm.ppf(1 - data), pvals.labels)

times = np.linspace(-0.5, 1.5, 200)

info = mne.create_info(ch_names=zscores.labels[1].tolist(), sfreq=100)
events = np.array([[0 + 200 * i, 0, i] for i in range(7)])
event_id = dict(zip(zscores.labels[0], range(7)))
picked = mne.time_frequency.EpochsTFRArray(
    info, sigs.__array__(), times, zscores.labels[2],
    events=events, event_id=event_id, drop_log=tuple(() for _ in range(7)))

# picked.average().plot(picks=0)
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
# plot_on_average(subjects, picks=picks, hemi='both')
