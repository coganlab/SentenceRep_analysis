## Description: Produce spectrograms for each subject
import mne.time_frequency

from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, crop_empty_data, outliers_to_nan
import os
from ieeg.timefreq.utils import crop_pad
import numpy as np

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
    subject = 7

def resample_tfr(tfr, sfreq, copy=False):
    """Resample a TFR object to a new sampling frequency"""
    if copy:
        tfr = tfr.copy()
    sfreq = int(sfreq)
    o_sfreq = tfr.info["sfreq"]

    tfr._data = downsample(tfr._data, sfreq, o_sfreq)
    lowpass = tfr.info.get("lowpass")
    lowpass = np.inf if lowpass is None else lowpass
    with tfr.info._unlock():
        tfr.info["lowpass"] = min(lowpass, sfreq / 2.0)
        tfr.info["sfreq"] = float(sfreq)
    new_times = downsample(tfr.times, sfreq, o_sfreq)
    # adjust indirectly affected variables
    tfr._set_times(new_times)
    tfr._raw_times = tfr.times
    tfr._update_first_last()
    return tfr

def downsample(arr: np.ndarray, new_sfreq: int, sfreq: int = None):
    """Downsample an array"""
    if sfreq is None:
        sfreq = arr.shape[-1]
    elif sfreq != arr.shape[-1]:
        factor = arr.shape[-1] / sfreq
        sfreq = int(round(sfreq * factor))
        new_sfreq = int(round(new_sfreq * factor))

    if new_sfreq > sfreq:
        raise ValueError("New sampling frequency must be less than original")
    elif new_sfreq == sfreq:
        return arr
    elif sfreq % new_sfreq != 0:
        indices = np.linspace(0, sfreq - 1, new_sfreq)
        indices = np.round(indices).astype(int)
        return arr[..., indices]
    else:
        return arr[..., ::sfreq // new_sfreq]

n_jobs = -2

for sub in subjects:
    # if int(sub[1:]) != subject:
    #     continue
    # Load the data
    filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                           extension='.edf', desc='notch', preload=False)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt,).copy()

    # good.info['bads'] = channel_outlier_marker(good, 3, 2)
    good.drop_channels(good.info['bads'])
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
        outliers_to_nan(trials, outliers=10)
        freq = np.geomspace(60, 300, num=40)
        kwargs = dict(average=False, n_jobs=n_jobs, freqs=freq, return_itc=False,
                      n_cycles=freq / 2, time_bandwidth=4)

        spec = trials.compute_tfr(method="multitaper", **kwargs)
        del trials
        if spec.sfreq > 100:
            resample_tfr(spec, 100)
        crop_pad(spec, "0.5s")
        # if epoch == "Start":
        #     base = spec.copy().crop(-0.5, 0)
        # spec_a = rescale(spec, base, copy=True, mode='zscore')
        # spec_a._data = np.log10(spec_a._data) * 20
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        spec.info['subject_info']['files'] = tuple(fnames)
        spec.info['bads'] = good.info['bads']
        filename = os.path.join(save_dir, f'{name}-tfr.h5')
        mne.time_frequency.write_tfrs(filename, spec, overwrite=True)
