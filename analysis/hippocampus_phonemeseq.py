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
    RECON_root = os.path.join(HOME, "Box", "ECoG_Recon")
layout = get_data("Phoneme_sequencing", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")
DData_dir = os.path.join(LAB_root, 'D_Data', 'Phoneme_Sequencing')
mat_fname = 'muscleChannelWavelet.mat'
analysisfolder = 'SentenceRep_analysis\\analysis'

# # %% Inspect raw/clean timeseries and save mt spectrogram to derivatives folder
# from ieeg.navigate import trial_ieeg, outliers_to_nan, channel_outlier_marker, crop_empty_data
# from ieeg.timefreq.utils import crop_pad
# from ieeg.viz.ensemble import chan_grid
# from ieeg.calc.scaling import rescale
# from ieeg.viz.parula import parula_map
# from analysis.check.chan_utils import get_muscle_chans
# #import mne
# import numpy as np
# import matplotlib.pyplot as plt
# #plt.switch_backend('Qt5Agg')
# save_dir = os.path.join(layout.root, "derivatives", "figs")
#
# for subj in subjects:
#     if int(subj[1:]) in [19,22,23,25,31,35,76,100]:
#         continue
#     # raw = raw_from_layout(layout, subject=subj, extension=".edf",
#     #                       desc=None, preload=True)
#     # raw = crop_empty_data(raw, ).copy()
#     filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
#                            extension='.edf', desc='clean', preload=True)
#     if "Trigger" in filt.ch_names:
#         filt.drop_channels(["Trigger"])
#     # Exclude muscle channels
#     muscle_chans = get_muscle_chans(DData_dir, mat_fname, subj)
#     filt = filt.drop_channels([chan for chan in muscle_chans if chan in filt.ch_names])
#     # Exclude outlier channels
#     filt.info['bads'] = channel_outlier_marker(filt, 3)
#     good = filt.drop_channels(filt.info['bads'])
#     del filt
#     # Exclude non-seeg channels
#     nonseeg_idx = [i for i, chtype in enumerate(good.get_channel_types()) if chtype not in ['seeg', 'ecog']]
#     good = good.drop_channels([good.ch_names[i] for i in nonseeg_idx])
#     good.load_data()
#     # CAR
#     ch_type = good.get_channel_types(only_data_chs=True)[0] #this is to account for some subj electrode coded as seeg vs. ecog (few, D24, 67?)
#     good.set_eeg_reference(ref_channels="average", ch_type=ch_type)
#     #good.plot()
#
#     base = trial_ieeg(good, 'Listen', (-1, 0.5), preload=True)
#     outliers_to_nan(base, outliers=10)
#     base_spectra = base.compute_tfr(method="multitaper", **kwargs)
#     # spectrogram with baseline via multitaper
#     for cond in ['Audio', 'Go', 'Response']:
#         freq = np.arange(10, 200., 6.)
#         trials = trial_ieeg(good, cond, (-1,1.5), preload=True)
#         outliers_to_nan(trials, outliers=10)
#         kwargs = dict(average=False, n_jobs=20, freqs=freq, return_itc=False,
#                       n_cycles=freq / 2, time_bandwidth=10,
#                       # n_fft=int(trials.info['sfreq'] * 2.75),
#                       decim=20, )
#         spectra = trials.compute_tfr(method="multitaper", **kwargs)
#         del trials
#         base = base_spectra.crop(-0.5, 0).average(lambda x: np.nanmean(x, axis=0), copy=True)
#         spectra = spectra.average(lambda x: np.nanmean(x, axis=0), copy=True)
#         rescale(spectra._data, base._data, mode='ratio', axis=-1)
#         crop_pad(spectra, "0.5s")
#         figs = chan_grid(spectra, vlim=(0.7, 1.4), cmap=parula_map, show=False)
#         for idx, fig in enumerate(figs):
#             fig.savefig(f"{save_dir}\\{subj}_{cond}_mt_{idx+1}.png")
#             plt.close(fig)
#
# #%% View power spectrum
# from ieeg.viz.ensemble import figure_compare
# for subj in subjects:
#     if int(subj[1:]) != 29:
#         continue
#     raw = raw_from_layout(layout, subject=subj, extension=".edf",
#                           desc=None, preload=True)
#     filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
#                            extension='.edf', desc='clean', preload=True)
#
#     figure_compare([raw, filt],
#                    labels=["Un", ""],
#                    avg=True,
#                    n_jobs=10,
#                    verbose=10,
#                    proj=True,
#                    fmax=250)
#
# #%% Morlet wavelet
#     # mask, pvals = time_perm_cluster(resp._data, base._data,
#     #                                 p_thresh=0.1,
#     #                                 ignore_adjacency=1,  # ignore channel adjacency
#     #                                 n_perm=2000, n_jobs=-1)
#     # fig, axs = plt.subplots(20, 10, figsize=(40, 20))
#     # for i, ax in enumerate(axs.flat):
#     #     if i >= mask.shape[0]:
#     #         ax.axis('off')
#     #         continue
#     #     ax.imshow(mask[i], aspect = 'auto')
#     #     ax.set_title(resp.info['ch_names'][i])
#     # plt.tight_layout()
#     # plt.show()
#
# # %% Fix event structure and line noise - Done
# from analysis.fix.events import fix
# from ieeg.mt_filter import line_filter
# from ieeg.io import get_data, raw_from_layout, save_derivative
# from ieeg.navigate import crop_empty_data
#
# for subj in subjects:
#     if int(subj[1:]) in [19,22,23,25,31,35,76]:
#         continue
#     else:
#         raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)
#         #DO NOT CROP before line_filter as additional run time is helpful for the line noise estimation
#         line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
#                 filter_length='700ms', freqs=[60], notch_widths=20)
#         line_filter(raw, mt_bandwidth=10., n_jobs=12, copy=False, verbose=10,
#                     filter_length='20s', freqs=[60, 120, 180, 240],
#                     notch_widths=20)
#
#     try:
#         fixed = fix(raw)
#         del raw
#         fixed = crop_empty_data(fixed)
#         # fixed.drop_channels('Trigger')
#         save_derivative(fixed, layout, "clean", True)
#     except Exception as e:
#         print(f"Error in {subj}: {e}")
#
# # %% Significant channels and save fifs
# from ieeg.navigate import trial_ieeg, outliers_to_nan, channel_outlier_marker
# from ieeg.calc.stats import time_perm_cluster
# from ieeg.calc import scaling
# from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
# import matplotlib.pyplot as plt
# from analysis.check.chan_utils import get_muscle_chans
# from ieeg.timefreq import gamma
# from mne import EvokedArray
# njobs = 4 #higher njobs can lead to overloading memory and slowing down the compute
# save_dir = os.path.join(layout.root, "derivatives", "stats")
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
#
# #load previously processed sigdict for updating
# # with open(f'{analysisfolder}\\sigdict_phonemeseq_small.pkl', 'rb') as f:
# #     sigdict = pickle.load(f)
#
# for subj in subjects:
#     if int(subj[1:]) in [19,22,23,25,31,35,76]:
#         continue
#     if int(subj[1:]) not in [102, 103]:
#         continue
#     filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
#                            extension=".edf", desc='clean', preload=False)
#     if "Trigger" in filt.ch_names:
#         filt.drop_channels(["Trigger"])
#     # Exclude muscle channels
#     muscle_chans = get_muscle_chans(DData_dir, mat_fname, subj)
#     filt = filt.drop_channels([chan for chan in muscle_chans if chan in filt.ch_names])
#     # Exclude outlier channels
#     filt.info['bads'] = channel_outlier_marker(filt, 3)
#     good = filt.drop_channels(filt.info['bads'])
#     # Exclude non-seeg channels
#     nonseeg_idx = [i for i, chtype in enumerate(good.get_channel_types()) if chtype not in ['seeg', 'ecog']]
#     good = good.drop_channels([good.ch_names[i] for i in nonseeg_idx])
#     good.load_data()
#     # CAR
#     ch_type = filt.get_channel_types(only_data_chs=True)[0] #this is to account for some subj electrode coded as seeg vs. ecog (few, D24, 67?)
#     good.set_eeg_reference(ref_channels="average", ch_type=ch_type)
#
#     # gamma extraction via Hilbert
#     out = []
#     for epoch, t in zip(('Listen', 'Audio', 'Go', 'Response'),  # epochs to extract
#                         ((-1, 0.5),(-0.5, 1),(-0.5, 1),(-0.5, 1))):  # times to extract
#         times = [None, None]
#         times[0] = t[0] - 0.5  # add 0.5s to the beginning
#         times[1] = t[1] + 0.5  # add 0.5s to the end
#         trials = trial_ieeg(good, epoch, times, preload=True)
#         # values greater than 10 standard deviations from the mean are set to NaN
#         # outliers_to_nan(trials, 10)
#         gamma.extract(trials, copy=False, n_jobs=njobs)
#         # trim 0.5 seconds on the beginning and end of the data (edge artifacts)
#         crop_pad(trials, "0.5s")
#         trials.resample(100) # downsample from 2kHz to 100Hz after extracting high gamma
#         out.append(trials)
#     base = out.pop(0)
#
#     # SIGDICT saving
#     # mask = []
#     # for i in range(len(out)):
#     #     if i == 0:
#     #         continue
#     #     mask[name], p_act = time_perm_cluster(out[i]._data, out[0]._data,
#     #                                    p_thresh=0.05,
#     #                                    axis=0,
#     #                                    n_perm=1000,
#     #                                    n_jobs=njobs,
#     #                                    ignore_adjacency=1)
#     #     mask.append(epoch_mask)
#     # sigdict[subj] = mask
#
#     # reduce size of sigdict by converting int32 to int8
#     # for key, value in sigdict.items():
#     #     sigdict[key] = [arr.astype(np.int8) for arr in value]
#     #
#     # with open(f'{analysisfolder}\\sigdict_phonemeseq_small.pkl', 'wb') as f:
#     #     pickle.dump(sigdict, f)
#
#     # time perm and save stats
#     mask = dict()
#     nperm = 5000
#     sig2 = base.get_data(copy=True)
#     for epoch, name, window in zip((out[0]["Audio"], out[1]["Go"], out[2]["Response"]),
#             ("aud", "go", "resp"),
#             ((-0.5, 1),(-0.5, 1),(-0.5, 1))):  # time-perm
#         sig1 = epoch.get_data(tmin=window[0], tmax=window[1], copy=True)
#
#         # time-perm
#         mask[name], p_act = time_perm_cluster(
#             sig1, sig2, p_thresh=0.05, axis=0, n_perm=nperm, n_jobs=njobs,
#             ignore_adjacency=1)
#         # build mne.EvokedArray to save the data with metadata
#         epoch_mask = EvokedArray(mask[name], epoch.average().info,
#                                      tmin=window[0])
#
#         power = scaling.rescale(epoch, base, 'mean', copy=True)
#         z_score = scaling.rescale(epoch, base, 'zscore', copy=True)
#
#         # Calculate the p-value
#         p_vals = EvokedArray(p_act, epoch_mask.info, tmin=window[0])
#
#         # saving epoch as fif file
#         power.save(save_dir + f"/{subj}_{name}_power-epo.fif", overwrite=True, fmt='double')
#         z_score.save(save_dir + f"/{subj}_{name}_zscore-epo.fif", overwrite=True, fmt='double')
#         epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.fif", overwrite=True)
#         p_vals.save(save_dir + f"/{subj}_{name}_pval-ave.fif", overwrite=True)
#     base.save(save_dir + f"/{subj}_base-epo.fif", overwrite=True)
#
#
# #%% check correct ch_type coding for seeg/ecog (D24, D67) vs. eeg - Done
# for subj in subjects:
#     if int(subj[1:]) in [19,22,23,25,31,35,76]:
#         continue
#     filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
#                            extension=".edf", desc='clean', preload=False)
#     eeg_ch_ind = [i for i, ch in enumerate(filt.ch_names) if len(ch)<4]
#     ch_types = filt.get_channel_types(only_data_chs=True)
#     eeg_ch_coding = [i for i, ch_type in enumerate(ch_types) if ch_type not in ['seeg','ecog']]
#     if eeg_ch_ind == eeg_ch_coding:
#         continue
#     else:
#         condition_fulfilled = False
#         while not condition_fulfilled:
#             # Wait for the user to press Enter (without typing anything)
#             user_input = input("Record subj number then press Enter to continue:")
#             if user_input == '':
#                 condition_fulfilled = True
#
# #%% read significant channels and saving sig channel plots
# import pickle
# with open('SentenceRep_analysis\\analysis\\sigdict_phonemeseq_small.pkl', 'rb') as f:
#     sigdict = pickle.load(f)
#
# figlabel = ["Audio", "Go", "Response"]
# for subj in subjects:
#     if subj in sigdict:
#         plt.figure()
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # (1 row, 3 columns)
#         # Loop through the list and plot each ndarray in a subplot
#         for i, ax in enumerate(axes):
#             ax.imshow(sigdict[subj][i], cmap='viridis')  # You can change the colormap
#             ax.set_title(figlabel[i])
#             ax.set_aspect('auto')
#         plt.savefig(f'SigChans_{subj}.png')
#         plt.close()
#
#%% save ch names
# To regenerate channel label and save as channel_label.pkl
from analysis.grouping import GroupData
from analysis.check.chan_utils import get_ch_label, get_preferred_ch_label

sub = GroupData.from_intermediates("Phoneme_sequencing", LAB_root, {'aud': (-0.5,1), 'go': (-0.5,1), 'resp': (-0.5,1)}, folder='stats', subjects_dir=RECON_root)
# get sig chans across 3 windows, across 150 timepoints
sub.sig_chans = np.where(np.any(sub.signif == 1, axis=(0,2)))[0].tolist()
sub_channels = sub.keys.get('channel',[])
orig_ch_label = get_ch_label(sub_channels, RECON_root)
maxhipp_ch_label = get_preferred_ch_label(sub_channels, RECON_root, "Hipp", 0.1) # at least 10% 10mm-radius sphere overlap with hippocampus
with open(f'{analysisfolder}\\sub_channel_phonemeseq.pkl', 'wb') as f:
    pickle.dump(sub_channels, f)
with open(f'{analysisfolder}\\channel_label_phonemeseq.pkl', 'wb') as f:
    pickle.dump(orig_ch_label, f)
with open(f'{analysisfolder}\\maxhipp_channel_label_phonemeseq.pkl', 'wb') as f:
    pickle.dump(maxhipp_ch_label, f)

#%%  plot
sub.plot_groups_on_average([idx_hippsig], colors = ['blue'])

#%% Prep data to labeledarray
from ieeg.calc.mat import LabeledArray, combine
from analysis.decoding import Decoder
from analysis.utils.mat_load import load_dict
from mne import read_epochs
import numpy as np

read_dir = os.path.join(layout.root, "derivatives", "stats")
suffix = "zscore-epo.fif"
conds = {'aud': (-0.5,1), 'go': (-0.5,1), 'resp': (-0.5,1)}

# zscores = dict()
# for subj in subjects:
#     if int(subj[1:]) in [19,22,23,25,31,35,76]:
#         continue
#     zscores[subj] = dict()
#     for cond in conds.keys():
#         try:
#             fname = os.path.join(read_dir, f"{subj}_{cond}_{suffix}")
#             zscores[subj] = LabeledArray.from_signal(read_epochs(fname))
#         except FileNotFoundError as e:
#             continue

zscores = load_dict(layout, conds, 'zscore', False, 'stats')
mask = load_dict(layout, conds, 'significance', True, 'stats')
zscores = combine(zscores, (0, 3)) # combine subj with channel
zscoresLA = LabeledArray.from_dict(zscores)
mask = combine(mask, (0, 2)) # combine subj with channel as stims already averaged
maskLA = LabeledArray.from_dict(mask)
sig_idx = dict()
# get sig chans for each cond
for i, cond in enumerate(conds.keys()):
    if cond not in sig_idx:
        sig_idx[cond] = {}
    sig_idx[cond] = np.where(np.any(maskLA.__array__()[i,:,:] == 1, axis = 1))[0].tolist()
# get union of sig chans
sig_idx_union = list(set(sig_idx['aud'])|set(sig_idx['go'])|set(sig_idx['resp']))
zscoresLA = zscoresLA.combine((1,3))

with open(f'{analysisfolder}\\sub_channel_phonemeseq.pkl', 'rb') as f:
    sub_channels = pickle.load(f)
with open(f'{analysisfolder}\\channel_label_phonemeseq.pkl', 'rb') as f:
    orig_ch_label = pickle.load(f)
with open(f'{analysisfolder}\\maxhipp_channel_label_phonemeseq.pkl', 'rb') as f:
    maxhipp_ch_label = pickle.load(f)
bad_words = ('Unknown', 'unknown', 'hypointensities', 'White-Matter', 'ERROR') # ERROR for elec unable to locate in RECON folder
idx_gm = [i for i, label in enumerate(orig_ch_label) if label not in bad_words]
idx_gmsig = [i for i in idx_gm if i in sig_idx_union]
idx_hipp = [i for i, label in enumerate(orig_ch_label) if 'Hipp' in label]
idx_hippsig = [i for i in idx_hipp if i in sig_idx_union]
idx_maxhipp = [i for i, label in enumerate(maxhipp_ch_label) if 'Hipp' in label]
idx_maxhippsig = [i for i in idx_maxhipp if i in sig_idx_union]
sig_idx_aud = list(set(sig_idx['aud']))
idx_gmsig_aud = [i for i in idx_gm if i in sig_idx_aud]
sig_idx_resp = list(set(sig_idx['resp']))
idx_gmsig_resp = [i for i in idx_gm if i in sig_idx_resp]

#%%test
from analysis.decoding import classes_from_labels
from analysis.check.chan_utils import remove_min_nan_ch, equal_valid_trials_ch, left_adjust_by_stim
import numpy as np

with open(f'{analysisfolder}\\zscores_test.pkl', 'rb') as f:
    zscores_test = pickle.load(f)

true_cat = {'abae':1, 'abi':1, 'aka':1, 'aku':1, 'ava':1, 'avae':1,
                 'aeba':2, 'aebi':2, 'aebu':2, 'aega':2, 'aeka':2, 'aepi':2,
                 'ibu':3, 'ika':3, 'ikae':3, 'ipu':3, 'iva':3, 'ivu':3,
                 'uba':4, 'uga':4, 'ugae':4, 'ukae':4, 'upi':4, 'upu':4, 'uvae':4, 'uvi':4,
                 'bab':5, 'baek':5, 'bak':5, 'bup':5,
                 'gab':6, 'gaeb':6, 'gaev':6, 'gak':6, 'gav':6, 'gig':6, 'gip':6, 'gub':6,
                 'kab':7, 'kaeg':7, 'kub':7, 'kug':7,
                 'paek':8, 'paep':8, 'paev':8, 'puk':8, 'pup':8,
                 'vaeg':9, 'vaek':9, 'vip':9, 'vug':9, 'vuk':9}
decoder_cat = {'a':1, 'ae':2, 'i':3, 'u':4, 'b':5, 'g':6, 'k':7, 'p':8, 'v':9}
cats, labels = classes_from_labels(zscores_test.labels[1], crop=slice(0, 4)) #this get out repetitions of same stims
flipped_cats = {v:k for k,v in cats.items()}
new_labels = np.array([true_cat[flipped_cats[l]] for l in labels]) #convert to true categories
zscoresLA_cond_idx, _ = remove_min_nan_ch(zscores_test, new_labels, min_non_nan=10)
zscoresLA_cond_idx_reduced = equal_valid_trials_ch(zscoresLA_cond_idx, new_labels, min_non_nan=10, upper_limit=10)


# %% Time sliding decoding for reconstruction
from analysis.decoding import classes_from_labels
from analysis.check.chan_utils import remove_min_nan_ch, equal_valid_trials_ch, left_adjust_by_stim, equal_stims_per_class

window_kwargs = {'obs_axs': 1, 'normalize': 'true', 'n_jobs': 10,
                'average_repetitions': False}
true_cat = {'abae':1, 'abi':1, 'aka':1, 'aku':1, 'ava':1, 'avae':1,
                 'aeba':2, 'aebi':2, 'aebu':2, 'aega':2, 'aeka':2, 'aepi':2,
                 'ibu':3, 'ika':3, 'ikae':3, 'ipu':3, 'iva':3, 'ivu':3,
                 'uba':4, 'uga':4, 'ugae':4, 'ukae':4, 'upi':4, 'upu':4, 'uvae':4, 'uvi':4,
                 'bab':5, 'baek':5, 'bak':5, 'bup':5,
                 'gab':6, 'gaeb':6, 'gaev':6, 'gak':6, 'gav':6, 'gig':6, 'gip':6, 'gub':6,
                 'kab':7, 'kaeg':7, 'kub':7, 'kug':7,
                 'paek':8, 'paep':8, 'paev':8, 'puk':8, 'pup':8,
                 'vaeg':9, 'vaek':9, 'vip':9, 'vug':9, 'vuk':9}
decoder_cat = {'a':1, 'ae':2, 'i':3, 'u':4, 'b':5, 'g':6, 'k':7, 'p':8, 'v':9}
iter_num = 10
for i_cond, cond in enumerate(conds.keys()):
    if i_cond != 0:
        continue
    zscoresLA_cond = zscoresLA.take(i_cond, axis=0)
    decoder = Decoder(decoder_cat, 0.8, 'lda', n_splits=5, n_repeats=10)
    cats, labels = classes_from_labels(zscoresLA.labels[2], crop=slice(0, 4)) #this get out repetitions of same stims
    flipped_cats = {v:k for k,v in cats.items()}
    new_labels = np.array([true_cat[flipped_cats[l]] for l in labels]) #convert to true categories
    zscoresLA_cond_idx = zscoresLA_cond.take(idx_hippsig, axis=0)
    zscoresLA_cond_idx = zscoresLA_cond_idx.take(np.arange(50)+49, axis = 2) #take only onset to 500ms
    zscoresLA_cond_idx, _ = remove_min_nan_ch(zscoresLA_cond_idx, new_labels, min_non_nan=15)
    scores_out = np.zeros((iter_num, len(decoder_cat), len(decoder_cat)), dtype = np.float16)
    for iter_i in range(iter_num):
        zscoresLA_cond_idx_reduced = equal_valid_trials_ch(zscoresLA_cond_idx, new_labels, min_non_nan=15, upper_limit=15)
        zscores_cropped, labels_cropped = left_adjust_by_stim(zscoresLA_cond_idx_reduced, new_labels, crop=True)
        scores_iter = decoder.cv_cm(zscores_cropped.__array__(), labels_cropped,
                                            **window_kwargs, oversample=False)
        scores_out[iter_i] = np.mean(scores_iter, axis=0)
    score_mean = np.mean(scores_out, axis=0)

#%% balanced cats
    scores_out = np.zeros((iter_num, len(decoder_cat), len(decoder_cat)), dtype = np.float16)
    for iter_i in range(iter_num):
        # ignore valid or not, just randomly select from trials to balance out class and use mixup
        zscoresLA_cond_idx_balanced, labels_balanced = equal_stims_per_class(zscoresLA_cond_idx, new_labels, keep_trials=16)
        zscoresLA_cond_idx_balanced, _ = remove_min_nan_ch(zscoresLA_cond_idx_balanced, labels_balanced, min_non_nan=3)
        zscores_cropped, labels_cropped = left_adjust_by_stim(zscoresLA_cond_idx_balanced, labels_balanced, crop=False)
        scores_iter = decoder.cv_cm(zscores_cropped.__array__(), labels_cropped,
                                    **window_kwargs, oversample=True)
        scores_out[iter_i] = np.mean(scores_iter, axis=0)
    score_mean = np.mean(scores_out, axis=0)

#%% plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
cax = ax.imshow(score_mean, cmap='viridis', vmin=0, vmax=0.4)

ax.set_xticks(np.arange(9))
ax.set_yticks(np.arange(9))
ax.set_xticklabels([list(decoder_cat.keys())[i] for i in range(9)])
ax.set_yticklabels([list(decoder_cat.keys())[i] for i in range(9)])
for i in range(score_mean.shape[0]):
    ax.text(i, i, f'{score_mean[i, i]:.2f}', ha='center', va='center', color='white', fontsize=12, weight='bold')
fig.colorbar(cax)
plt.tight_layout()
plt.show()