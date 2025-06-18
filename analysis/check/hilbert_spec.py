## Description: Produce spectrograms for each subject

from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, crop_empty_data, outliers_to_nan, find_bad_channels_lof
from ieeg.timefreq.gamma import hilbert_spectrogram
import os
from ieeg.timefreq.utils import crop_pad, resample_tfr
import numpy as np
import scipy.stats as st
import functools

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
    subject = 29

n_jobs = 1
for sub in subjects:
    if int(sub[1:]) in (32,):
        continue
    if subject is not None:
        if int(sub[1:]) != subject:
            continue
    # Load the data
    filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                           extension='.edf', desc='notch', preload=False)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt,).copy()

    good.info['bads'] = []
    # good.info['bads'] = channel_outlier_marker(good, 3, 2)
    bads = list(set(filt.info['bads']))
    good.drop_channels(bads)
    good.load_data()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    # scheme = pre.make_contact_rereference_arr(good.ch_names)
    # good._data = pre.rereference(scheme, field=[good._data.T])[0].T
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'hilbert', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch, t, name in zip(
            ("Start", "Word/Response/LS", "Word/Audio/LS", "Word/Audio/LM",
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
        func = functools.partial(st.iqr, rng=(50, 95), nan_policy='omit')
        outliers_to_nan(trials, outliers=6, deviation=func,
                        center=np.nanmedian, tmin=t[0], tmax=t[1])
        spec = hilbert_spectrogram(trials, (4, 500),4, 1/12, n_jobs)
        crop_pad(spec, "0.5s")
        outliers_to_nan(spec, outliers=6, deviation=func,
                        center=np.nanmedian)
        resample_tfr(spec, 100, spec.times.shape[0] / (spec.tmax - spec.tmin))
        # if spec.sfreq > 100:
        #     # factor = min(2, spec.sfreq // 100)
        #     # offset = len(spec.times) % 100
        #     # spec = spec.decimate(factor, offset)
        #     resample_tfr(spec, 100, spec.times.shape[0] / (spec.tmax - spec.tmin))
            # if name == "start":
            #     resample_tfr(spec, 100, spec.times.shape[0])
            # else:
            #     resample_tfr(spec, 200, spec.times.shape[0])
            #     resample_tfr(spec, 100, 100)

        # if epoch == "Start":
        #     base = spec.copy().crop(-0.5, 0)
        # spec_a = rescale(spec, base, copy=True, mode='zscore')
        # spec_a._data = np.log10(spec_a._data) * 20
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        #spec.info['subject_info']['files'] = tuple(fnames)
        # spec.info['bads'] = bads
        filename = os.path.join(save_dir, f'{name}-tfr.h5')
        # mne.time_frequency.write_tfrs(filename, spec, overwrite=True)

        spec.save(filename, overwrite=True, verbose=True)
