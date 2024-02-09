## Description: Produce spectrograms for each subject
import mne.time_frequency

from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data,\
    outliers_to_nan
from ieeg.calc.scaling import rescale
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
from ieeg.timefreq.multitaper import spectrogram
import numpy as np
import naplib.preprocessing as pre

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
    # Load the data
    filt = raw_from_layout(layout.derivatives['clean'], subject=sub,
                           extension='.edf', desc='clean', preload=False)

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

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'multitaper', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch, t, name in zip(
            ("Start", "Start", "Word/Response/LS", "Word/Audio/LS", "Word/Audio/LM",
             "Word/Audio/JL", "Word/Go/LS", "Word/Go/LM",
             "Word/Go/JL"),
            ((-0.5, 0), (-0.5, 0.5), (-1, 1), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5),
             (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)),
            ("base", "start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm",
             "go_jl")):
        t1 = t[0] - 0.5
        t2 = t[1] + 0.5
        trials = trial_ieeg(good, epoch, (t1, t2), preload=True)
        outliers_to_nan(trials, outliers=20)
        if name == "base":
            base = trials.copy()
            continue
        freq = np.arange(10, 200., 2.)
        spectra = spectrogram(trials, freq, base, n_jobs=6, pad="0.5s",
                              n_cycles=freq / 2, time_bandwidth=20)
        spec_a = spectra.copy()
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        spec_a.info['subject_info']['files'] = tuple(fnames)
        spec_a.info['bads'] = good.info['bads']
        filename = os.path.join(save_dir, f'{name}-tfr.h5')
        mne.time_frequency.write_tfrs(filename, spec_a, overwrite=True)