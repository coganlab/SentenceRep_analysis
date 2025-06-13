from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, crop_empty_data, outliers_to_nan, channel_outlier_marker, find_bad_channels_lof
from ieeg.timefreq.gamma import hilbert_spectrogram
import os
from ieeg.timefreq.utils import crop_pad, resample_tfr
import ieeg.viz
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.neighbors import LocalOutlierFactor
import mne

## check if currently running a slurm job
HOME = os.path.expanduser("~")

if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    subject = None

fig, axs = plt.subplots(7, 5, figsize=(20, 10))
for i, sub in enumerate(subjects):
    if int(sub[1:]) in (30, 32):
        continue
    if subject is not None:
        if int(sub[1:]) != subject:
            continue
    ax = axs[i // 5, i % 5]

    # Load the data
    filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                           extension='.edf', desc='notch', preload=False)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt,).copy()
    good.load_data()
    print(good.info['bads'])
    good.info['bads'] = []
    # channel_outlier_marker(good, 3, 2, save=False)
    channels, scores = find_bad_channels_lof(good,
                                             metric='seuclidean',
                                             return_scores=True,
                                             metric_params={'V': np.var(good._data, axis=0)},
                                             n_jobs=-1)
    print(channels)
    new_thresh = 1.5
    while np.mean(-scores > new_thresh, dtype=float) > 0.2:
        new_thresh += 1
    ax.plot(scores)
    ax.set_xticks([])
    # ax.set_xticks(np.arange(0, len(scores), 5))
    # ax.set_xticklabels([good.ch_names[i] for i in range(0, len(scores), 5)], rotation=45)
    ax.axhline(-1.5, color='r', linestyle='--')
    ax.set_title(sub)
fig.tight_layout()