## Description: Produce spectrograms for each subject
import mne.time_frequency

from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, crop_empty_data, outliers_to_nan
from ieeg.calc.oversample import resample
from ieeg.timefreq.gamma import hilbert_spectrogram
import os
from ieeg.timefreq.utils import crop_pad, cwt
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
    subject = 32

def resample_tfr(tfr, sfreq, o_sfreq=None, copy=False):
    """Resample a TFR object to a new sampling frequency"""
    if copy:
        tfr = tfr.copy()

    if o_sfreq is None:
        # o_sfreq = len(tfr.times) / (tfr.tmax - tfr.tmin)
        o_sfreq = tfr.info["sfreq"]

    tfr._data = resample(tfr._data, o_sfreq, sfreq, axis=-1)
    lowpass = tfr.info.get("lowpass")
    lowpass = np.inf if lowpass is None else lowpass
    with tfr.info._unlock():
        tfr.info["lowpass"] = min(lowpass, sfreq / 2)
        tfr.info["sfreq"] = sfreq
    new_times = resample(tfr.times, o_sfreq, sfreq, axis=-1)
    # adjust indirectly affected variables
    tfr._set_times(new_times)
    tfr._raw_times = tfr.times
    tfr._update_first_last()
    return tfr

n_jobs = 7
for sub in subjects:
    if int(sub[1:]) in (30, 32):
        continue
    # Load the data
    filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                           extension='.edf', desc='notch', preload=False)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt,).copy()

    # good.info['bads'] = channel_outlier_marker(good, 3, 2)
    bads = good.info['bads']
    good.drop_channels(bads)
    good.load_data()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    # scheme = pre.make_contact_rereference_arr(good.ch_names)
    # good._data = pre.rereference(scheme, field=[good._data.T])[0].T
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'multitaper', sub)
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
        freq = np.linspace(10, 500, num=25)
        time_smooth = 0.3 #seconds
        bw = (freq[1] - freq[0]) * time_smooth
        # decim = int(trials.info['sfreq']) // 100
        kwargs = dict(average=False, n_jobs=n_jobs, freqs=freq, return_itc=False,
                      n_cycles=freq * time_smooth, time_bandwidth=11,
                      decim=4)
        spec = trials.compute_tfr(method="multitaper", **kwargs)
        # spec = cwt(trials, 10, 500, 8, 4, 40)
        # spec = hilbert_spectrogram(trials, (30, 500), 4, 1/8.7, n_jobs)
        crop_pad(spec, "0.5s")
        # spec = spec.decimate(2, 1)
        del trials

        # if spec.sfreq % 100 == 0:
        #     factor = spec.sfreq // 100
        #     offset = len(spec.times) % factor
        #     spec = spec.decimate(factor, offset)
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
