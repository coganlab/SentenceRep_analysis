from timeit import timeit
import mne
from bids import BIDSLayout
from ieeg.io import raw_from_layout

bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
raw = raw_from_layout(layout, subject="pt1", preload=True, extension = ".vhdr", verbose = False)
events, ids = mne.events_from_annotations(raw)
trials = mne.Epochs(raw, events, event_id=ids, tmin=-10, tmax=10, preload=True, verbose=False)
ids_rev = {v: k for k, v in ids.items()}
trials.drop_bad()


def iter_over_ch_ev(trials: mne.Epochs):
    for ch in trials.ch_names:
        for ev in events[:, 2]:
            return trials.get_data([ch], item=ids_rev[ev])


def iter_over_array(trials):
    arr = trials.get_data()
    for i in range(arr.shape[1]):
        for j in range(arr.shape[0]):
            return arr[j, i]

# %timeit iter_over_ch_ev(trials)
# %timeit iter_over_array(trials)
