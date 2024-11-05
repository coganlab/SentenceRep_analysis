# %% Import dir
import os
from ieeg.io import get_data, raw_from_layout

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("Phoneme_sequencing", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")
matdir = os.path.join(LAB_root, 'D_Data', 'Phoneme_Sequencing')
matfname = 'muscleChannelWavelet.mat'

# %% Inspect raw/clean timeseries
from ieeg.navigate import channel_outlier_marker, crop_empty_data
import mne
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

for subj in subjects:
    if int(subj[1:]) != 61:
        continue
    raw = raw_from_layout(layout, subject=subj, extension=".edf",
                          desc=None, preload=True)
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension='.edf', desc='clean', preload=True)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(raw, ).copy()

    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    good.drop_channels(good.info['bads'])
    good.load_data()
    good.plot()

# spectrogram with baseline via Fourier
    # freq = np.arange(10, 200., 6.)
    # base = trial_ieeg(good, 'Listen', (-1, 0.5), preload=True)
    # outliers_to_nan(base, outliers=10)
    # trials = trial_ieeg(good, 'Response', (-1.2, 1.2), preload=True)
    # outliers_to_nan(trials, outliers=10)
    #
    # kwargs = dict(average=False, n_jobs=20, freqs=freq, return_itc=False,
    #               n_cycles=freq / 2, time_bandwidth=10,
    #               # n_fft=int(trials.info['sfreq'] * 2.75),
    #               decim=20, )
    #
    # spectra = trials.compute_tfr(method="multitaper", **kwargs)
    # base_spectra = base.compute_tfr(method="multitaper", **kwargs)
    # del trials
    # base = base_spectra.crop(-0.5, 0).average(lambda x: np.nanmean(x, axis=0), copy=True)
    #
    # spectra = spectra.average(lambda x: np.nanmean(x, axis=0), copy=True)
    # rescale(spectra._data, base._data, mode='ratio', axis=-1)
    # crop_pad(spectra, "0.5s")
    # chan_grid(spectra, vlim=(0.7, 1.4), cmap=parula_map)

# %% Fix event structure and line noise
from analysis.fix.events import fix
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.navigate import crop_empty_data

for subj in subjects:
    if (int(subj[1:]) in [19,22,23,25,31,35]):
        continue
    else:
        raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)
        #DO NOT CROP before line_filter as additional run time is helpful for the line noise estimation
        line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
                filter_length='700ms', freqs=[60], notch_widths=20)
        # line_filter(raw, mt_bandwidth=10., n_jobs=12, copy=False, verbose=10,
        #             filter_length='20s', freqs=[60, 120, 180, 240],
        #             notch_widths=20)

    try:
        fixed = fix(raw)
        del raw
        fixed = crop_empty_data(fixed)
        # fixed.drop_channels('Trigger')
        save_derivative(fixed, layout, "clean", True)
    except Exception as e:
        print(f"Error in {subj}: {e}")



# %% Significant channels
from ieeg.navigate import trial_ieeg, outliers_to_nan, crop_empty_data
from ieeg.calc.stats import time_perm_cluster
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import matplotlib.pyplot as plt
from analysis.check.chan_utils import get_muscle_chans
from ieeg.timefreq import gamma

for subj in subjects:
    if (int(subj[1:]) != 61):
            #in [19, 22, 23, 25, 31]):
        continue
    else:
        filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                               extension=".edf", desc='clean', preload=False)
        if "Trigger" in filt.ch_names:
            filt.drop_channels(["Trigger"])
        # Exclude bad channels
        good = filt.drop_channels(filt.info['bads'])
        good = good.drop_channels(get_muscle_chans(matdir, matfname, subj))
        good.load_data()
        # CAR
        ch_type = good.get_channel_types(only_data_chs=True)[0]
        good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    # Morlet wavelet calculation for gamma range
    out = []
    for epoch, t in zip(('Listen', 'Response'),  # epochs to extract
                        ((-1, 0.5),(-0.5, 1))):  # times to extract
        times = [None, None]
        times[0] = t[0] - 0.5  # add 0.5s to the beginning
        times[1] = t[1] + 0.5  # add 0.5s to the end
        trials = trial_ieeg(good, epoch, times, preload=True)
        # values greater than 10 standard deviations from the mean are set to NaN
        outliers_to_nan(trials, 10)
        gamma.extract(trials, copy=False, n_jobs=1)
        # calculate morlet wavelet transform in EpochTFR out format
        spec = wavelet_scaleogram(trials,
                                  n_jobs=-1,
                                  decim=20)
        # trim 0.5 seconds on the beginning and end of the data (edge artifacts)
        crop_pad(spec, "0.5s")
        out.append(spec)
    resp = out[1]
    base = out[0]

    mask, pvals = time_perm_cluster(resp._data, base._data,
                                    p_thresh=0.1,
                                    ignore_adjacency=1,  # ignore channel adjacency
                                    n_perm=2000, n_jobs=-1)
    fig, axs = plt.subplots(20, 10, figsize=(40, 20))
    for i, ax in enumerate(axs.flat):
        if i >= mask.shape[0]:
            ax.axis('off')
            continue
        ax.imshow(mask[i], aspect = 'auto')
        ax.set_title(resp.info['ch_names'][i])
    plt.tight_layout()
    plt.show()