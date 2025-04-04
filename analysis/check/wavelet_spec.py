## Description: Produce spectrograms for each subject
import mne.time_frequency

from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, find_bad_channels_lof,\
    outliers_to_nan, crop_empty_data
import scipy.stats as st
import os
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram, resample_tfr
import numpy as np

## check if currently running a slurm job
HOME = os.path.expanduser("~")

if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = list(int(os.environ['SLURM_ARRAY_TASK_ID']))
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")

for sub in subjects:
    if sub == "D0032":
        continue

    # Load the data
    filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                           extension='.edf', desc='notch', preload=False)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt,).copy()

    good.info['bads'] = find_bad_channels_lof(good, n_jobs=-1)
    good.drop_channels(good.info['bads'])
    good.load_data()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)


    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch, t, name in zip(
            ("Start",  "Word/Response/LS", "Word/Audio/LS", "Word/Audio/LM",
             "Word/Audio/JL", "Word/Go/LS", "Word/Go/LM",
             "Word/Go/JL"),
            ((-0.5, 0.5), (-1, 1), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5),
             (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)),
            ("start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm",
             "go_jl")):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True)
        outliers_to_nan(trials, outliers=30, deviation=st.median_abs_deviation, center=np.median)
        spec = wavelet_scaleogram(trials, n_jobs=-2, decim=4)
        crop_pad(spec, "0.5s")
        resample_tfr(spec, 100, spec.times.shape[0] / (spec.tmax - spec.tmin))
        if epoch == "Start":
            base = spec.copy().crop(-0.5, 0)

        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        filename = os.path.join(save_dir, f'{name}-tfr.h5')
        spec.save(filename, overwrite=True, verbose=True)