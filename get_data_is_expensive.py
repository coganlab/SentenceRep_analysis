from timeit import timeit
import mne
from mne_bids import BIDSPath, read_raw_bids


bids_path = BIDSPath(
    root=mne.datasets.epilepsy_ecog.data_path(),
    subject="pt1",
    session="presurgery",
    task="ictal",
    datatype="ieeg",
    extension=".vhdr",
)
raw = read_raw_bids(bids_path=bids_path, verbose="error")
events, ids = mne.events_from_annotations(raw)
trials = mne.Epochs(raw, events, event_id=ids, tmin=-10, tmax=10, preload=True, verbose=False)
ids_rev = {v: k for k, v in ids.items()}
trials.drop_bad()


def iter_over_ch_ev():
    for ch in trials.ch_names:
        for ev in events[:, 2]:
            yield trials.get_data([ch], item=ids_rev[ev])


def iter_over_array():
    arr = trials.get_data()
    for i in range(arr.shape[1]):
        for j in range(arr.shape[0]):
            yield arr[j, i]


ch_ev = timeit(lambda: list(iter_over_ch_ev()), number=10)
arr = timeit(lambda: list(iter_over_array()), number=10)

print(f'iterating over mne object: {ch_ev:.4}s')
print(f'iterating over array: {arr:.4}s')
print(f'array is {ch_ev / arr:.2f}x faster')
