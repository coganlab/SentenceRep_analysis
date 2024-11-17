# %% Import dir
import os
from ieeg.io import get_data, raw_from_layout
import pickle

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
analysisfolder = 'SentenceRep_analysis\\analysis'

# %% Inspect raw/clean timeseries
from ieeg.navigate import channel_outlier_marker, crop_empty_data
import mne
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

for subj in subjects:
    if int(subj[1:]) != 100:
        continue
    # raw = raw_from_layout(layout, subject=subj, extension=".edf",
    #                       desc=None, preload=True)
    #good = crop_empty_data(raw, ).copy()
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension='.edf', desc='clean', preload=True)
    good = filt.copy()
    del filt

    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    good.drop_channels(good.info['bads'])
    good.load_data()
    good.plot()

# spectrogram with baseline via multitaper
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

# %% Fix event structure and line noise - Done
from analysis.fix.events import fix
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.navigate import crop_empty_data

for subj in subjects:
    if int(subj[1:]) in [19,22,23,25,31,35,76]:
        continue
    else:
        raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)
        #DO NOT CROP before line_filter as additional run time is helpful for the line noise estimation
        line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
                filter_length='700ms', freqs=[60], notch_widths=20)
        line_filter(raw, mt_bandwidth=10., n_jobs=12, copy=False, verbose=10,
                    filter_length='20s', freqs=[60, 120, 180, 240],
                    notch_widths=20)

    try:
        fixed = fix(raw)
        del raw
        fixed = crop_empty_data(fixed)
        # fixed.drop_channels('Trigger')
        save_derivative(fixed, layout, "clean", True)
    except Exception as e:
        print(f"Error in {subj}: {e}")

# %% Significant channels
from ieeg.navigate import trial_ieeg, outliers_to_nan, channel_outlier_marker
from ieeg.calc.stats import time_perm_cluster
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import matplotlib.pyplot as plt
import numpy as np
from analysis.check.chan_utils import get_muscle_chans
from ieeg.timefreq import gamma
import pickle
njobs = 4 #higher njobs can lead to overloading memory and slowing down the compute

#load previously processed sigdict for updating if necessary
with open(f'{analysisfolder}\\sigdict_phonemeseq_small.pkl', 'rb') as f:
    sigdict = pickle.load(f)

for subj in subjects:
    if int(subj[1:]) in [19,22,23,25,31,35,76]:
        continue
    if int(subj[1:]) != 95:
        continue
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension=".edf", desc='clean', preload=False)
    if "Trigger" in filt.ch_names:
        filt.drop_channels(["Trigger"])
    # Exclude bad channels
    muscle_chans = get_muscle_chans(matdir, matfname, subj)
    filt = filt.drop_channels([chan for chan in muscle_chans if chan in filt.ch_names])
    filt.info['bads'] = channel_outlier_marker(filt, 3)
    good = filt.drop_channels(filt.info['bads'])
    good.load_data()
    # CAR
    #good.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type="seeg")

    # gamma extraction via Hilbert
    out = []
    for epoch, t in zip(('Listen', 'Audio', 'Go', 'Response'),  # epochs to extract
                        ((-1, 0.5),(-0.5, 1),(-0.5, 1),(-0.5, 1))):  # times to extract
        times = [None, None]
        times[0] = t[0] - 0.5  # add 0.5s to the beginning
        times[1] = t[1] + 0.5  # add 0.5s to the end
        trials = trial_ieeg(good, epoch, times, preload=True)
        # values greater than 10 standard deviations from the mean are set to NaN
        # outliers_to_nan(trials, 10)
        gamma.extract(trials, copy=False, n_jobs=njobs)
        # trim 0.5 seconds on the beginning and end of the data (edge artifacts)
        crop_pad(trials, "0.5s")
        out.append(trials)

    mask = []
    for i in range(len(out)):
        if i == 0:
            continue
        epoch_mask, _ = time_perm_cluster(out[i]._data, out[0]._data,
                                       p_thresh=0.05,
                                       axis=0,
                                       n_perm=1000,
                                       n_jobs=njobs,
                                       ignore_adjacency=1)
        mask.append(epoch_mask)
    sigdict[subj] = mask

    # reduce size of sigdict by converting int32 to int8
    for key, value in sigdict.items():
        sigdict[key] = [arr.astype(np.int8) for arr in value]

    with open(f'{analysisfolder}\\sigdict_phonemeseq_small.pkl', 'wb') as f:
        pickle.dump(sigdict, f)

#%% check correct ch_type coding for seeg vs. eeg
for subj in subjects:
    if int(subj[1:]) in [19,22,23,25,31,35,76]:
        continue
    if int(subj[1:]) != 102:
        continue
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension=".edf", desc='clean', preload=False)
    eeg_ch_ind = [i for i, ch in enumerate(filt.ch_names) if len(ch)<4]
    ch_types = filt.get_channel_types(only_data_chs=True)
    eeg_ch_coding = [i for i, ch in enumerate(ch_types) if ch != 'seeg']
    if eeg_ch_ind == eeg_ch_coding:
        continue
    else:
        condition_fulfilled = False
        while not condition_fulfilled:
            # Wait for the user to press Enter (without typing anything)
            user_input = input("Record subj number then press Enter to continue:")
            if user_input == '':
                condition_fulfilled = True


#%% read significant channels and saving plots
import pickle
with open('SentenceRep_analysis\\analysis\\sigdict_phonemeseq_small.pkl', 'rb') as f:
    sigdict = pickle.load(f)

figlabel = ["Audio", "Go", "Response"]
for subj in subjects:
    if int(subj[1:]) != 100:
        continue
    if subj in sigdict:
        plt.figure()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # (1 row, 3 columns)
        # Loop through the list and plot each ndarray in a subplot
        for i, ax in enumerate(axes):
            ax.imshow(sigdict[subj][i], cmap='viridis')  # You can change the colormap
            ax.set_title(figlabel[i])
            ax.set_aspect('auto')
        plt.savefig(f'SigChans_{subj}_2.png')
        plt.close()



#%% Morlet wavelet
    # mask, pvals = time_perm_cluster(resp._data, base._data,
    #                                 p_thresh=0.1,
    #                                 ignore_adjacency=1,  # ignore channel adjacency
    #                                 n_perm=2000, n_jobs=-1)
    # fig, axs = plt.subplots(20, 10, figsize=(40, 20))
    # for i, ax in enumerate(axs.flat):
    #     if i >= mask.shape[0]:
    #         ax.axis('off')
    #         continue
    #     ax.imshow(mask[i], aspect = 'auto')
    #     ax.set_title(resp.info['ch_names'][i])
    # plt.tight_layout()
    # plt.show()


#%% from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from ieeg.navigate import channel_outlier_marker, trial_ieeg, outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.mt_filter import line_filter
import mne
import numpy as np
import matplotlib.pyplot as plt

