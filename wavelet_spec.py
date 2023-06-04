## Description: Produce spectrograms for each subject
import mne.time_frequency

from ieeg.io import get_data, raw_from_layout, update, save_derivative
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data,\
    outliers_to_nan
from ieeg.calc.scaling import rescale
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
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
    # Load the data
    filt = raw_from_layout(layout.derivatives['clean'], subject=sub,
                           extension='.edf', desc='clean', preload=False)

    ## fix SentenceRep events
    from events import fix_annotations  # noqa E402
    new = crop_empty_data(filt,)

    good = new.copy()
    fix_annotations(good)

    ## Crop raw data to minimize processing time

    # good.drop_channels(good.info['bads'])
    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    good.drop_channels(good.info['bads'])
    # good.info['bads'] += channel_outlier_marker(good, 4, 2)
    # good.drop_channels(good.info['bads'])
    good.load_data()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    # Remove intermediates from mem
    del new
    # good.plot()

    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch, t, name in zip(
            ("Start",  "Word/Response/LS", "Word/Audio/LS", "Word/Audio/LM",
             "Word/Audio/JL", "Word/Speak/LS", "Word/Mime/LM",
             "Word/Audio/JL"),
            ((-0.5, 0), (-1, 1), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5),
             (-0.5, 1.5), (-0.5, 1.5), (1, 3)),
            ("base", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm",
             "go_jl")):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True)
        outliers_to_nan(trials, outliers=10)
        spec = wavelet_scaleogram(trials, n_jobs=-2, decim=int(
            good.info['sfreq'] / 100))
        crop_pad(spec, "0.5s")
        if epoch == "Start":
            base = spec.copy()
            continue
        spec_a = rescale(spec, base, copy=True, mode='ratio').average(
            lambda x: np.nanmean(x, axis=0), copy=True)
        spec_a._data = np.log10(spec_a._data) * 20
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        spec_a.info['subject_info']['files'] = tuple(fnames)
        spec_a.info['bads'] = good.info['bads']
        filename = os.path.join(save_dir, f'{name}-tfr.h5')
        mne.time_frequency.write_tfrs(filename, spec_a, overwrite=True)
        # spec_a.save(os.path.join(save_dir, f'{name}-avg.fif'), overwrite=True)