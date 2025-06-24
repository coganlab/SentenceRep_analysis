# %% Import dir
import os
import numpy as np
from matplotlib import pyplot as plt
from ieeg.io import get_data

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
analysisfolder = 'SentenceRep_analysis\\analysis'

#%% proper bipolar reference - not used
# from analysis.fix.events import fix
from ieeg.mt_filter import line_filter
# from ieeg.io import raw_from_layout, save_derivative
# from ieeg.navigate import crop_empty_data
# from analysis.check.chan_utils import bipolar_reference_by_shank
# plt.switch_backend('Qt5Agg')
#
# failed_subjects = []
#
# for subj in subjects:
#     # if int(subj[1:]) in [19,22,23,25,31,35,76]:
#     #     continue
#     try:
#         raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)
#         bipolar = bipolar_reference_by_shank(raw)
#         #DO NOT CROP before line_filter as additional run time is helpful for the line noise estimation
#         line_filter(bipolar, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
#                 filter_length='700ms', freqs=[60], notch_widths=20)
#         line_filter(bipolar, mt_bandwidth=10., n_jobs=12, copy=False, verbose=10,
#                     filter_length='20s', freqs=[60, 120, 180, 240],
#                     notch_widths=20)
#         try:
#             fixed = fix(bipolar)
#             del raw, bipolar
#             fixed = crop_empty_data(fixed)
#             # fixed.drop_channels('Trigger')
#             save_derivative(fixed, layout, "bipolarclean", True)
#         except Exception as e:
#             print(f"Error in postprocessing for {subj}: {e}")
#             failed_subjects.append(subj)
#
#     except Exception as e:
#         print(f"Error in loading for {subj}: {e}")
#         failed_subjects.append(subj)


#%% save re-referenced ROI electrodes and get HG/theta
from ieeg.navigate import trial_ieeg, channel_outlier_marker, outliers_to_nan
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc import scaling
from ieeg.viz.ensemble import chan_grid
from ieeg.io import raw_from_layout, save_derivative
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram
from analysis.check.chan_utils import get_muscle_chans, get_ch_label, parse_channel_info, build_shank_map
from ieeg.timefreq import gamma
from ieeg.viz.parula import parula_map
from mne import EvokedArray, EpochsArray
from mne.filter import filter_data
njobs = 4 #higher njobs can lead to overloading memory and slowing down the compute
save_dir = os.path.join(layout.root, "derivatives", "lingrerefgammastats")
fig_dir = os.path.join(layout.root, "derivatives", "figs", "lingreref")
get_band = 'gamma'
if get_band == 'theta':
    pad_len = 1
else:
    pad_len = 0.5
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
no_roi_subj = []
for subj in subjects:
    if int(subj[1:]) in [19,22,23,25,31,35,76,100]:
        continue
    if int(subj[1:]) < 40:
        continue
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension=".edf", desc='clean', preload=False)
    if "Trigger" in filt.ch_names:
        filt.drop_channels(["Trigger"])
    # Exclude muscle channels
    muscle_chans = get_muscle_chans(DData_dir, 'muscleChannelWavelet.mat', subj)
    filt = filt.drop_channels([chan for chan in muscle_chans if chan in filt.ch_names])
    # Exclude outlier channels
    filt.info['bads'] = channel_outlier_marker(filt, 3)
    good = filt.drop_channels(filt.info['bads'])
    # Exclude non-seeg channels
    nonseeg_idx = [i for i, chtype in enumerate(good.get_channel_types()) if chtype not in ['seeg', 'ecog']]
    good = good.drop_channels([good.ch_names[i] for i in nonseeg_idx])
    good.load_data()

    # Prep bipolar referencing
    # Get channel labels
    sub_ch_list = [f'D{subj[1:]}-{ch}' for ch in good.ch_names]
    channel_labels = get_ch_label(sub_ch_list, RECON_root)
    # Build shank map for finding nearest references
    shank_map = build_shank_map(good.ch_names)

    # Track ROI channels and their referenced data
    roi_channels = []
    roi_data = []
    data = good.get_data()
    ROI_NAME = "LinG"

    for i, (ch, label) in enumerate(zip(good.ch_names, channel_labels)):
        if ROI_NAME in label:
            # Find reference electrode
            _, shank, pos = parse_channel_info(ch)
            pos_list = shank_map[None,shank]  # This should be a list of (position, index) tuples, None since within subj
            current_idx = next(j for j, (pos_j, idx) in enumerate(pos_list) if idx == i)

            # Search for nearest non-ROI reference
            ref_idx = None
            min_dist = float('inf')

            # Search up
            for j in range(current_idx - 1, -1, -1):
                pos_j, idx = pos_list[j]  # Unpack position and index
                if ROI_NAME not in channel_labels[idx]:
                    dist = abs(pos - pos_j)  # Compare positions directly
                    if dist < min_dist:
                        min_dist = dist
                        ref_idx = idx
                    break

            # Search down
            for j in range(current_idx + 1, len(pos_list)):
                pos_j, idx = pos_list[j]  # Unpack position and index
                if ROI_NAME not in channel_labels[idx]:
                    dist = abs(pos_j - pos)  # Compare positions directly
                    if dist < min_dist:
                        min_dist = dist
                        ref_idx = idx
                    break

            # If reference found, store channel and referenced data
            if ref_idx is not None:
                roi_channels.append(ch)
                roi_data.append(data[i] - data[ref_idx])
            else:
                print(f"No reference found for channel {ch}, dropped.")

    # Keep only ROI channels with their referenced data
    if roi_channels and len(roi_channels) > 0:
        good.pick_channels(roi_channels)
        good._data = np.array(roi_data)
    else:
        print(f"No {ROI_NAME} channels found for subject {subj}")
        no_roi_subj.append(subj)
        continue

    # gamma extraction via filterbank Hilbert
    out = []
    for epoch, t in zip(('Listen', 'Audio', 'Go', 'Response'),  # epochs to extract
                        ((-1, 0.5),(-0.5, 1),(-0.5, 1),(-0.5, 1))):  # times to extract
        times = [None, None]
        times[0] = t[0] - pad_len  # add 0.5s to the beginning, 1s for theta
        times[1] = t[1] + pad_len  # add 0.5s to the end, 1s for theta
        trials = trial_ieeg(good, epoch, times, preload=True)
        # values greater than 10 standard deviations from the mean are set to NaN
        outliers_to_nan(trials, outliers=10)
        # Get Morlet spectrograms and save
        spec = wavelet_scaleogram(trials, n_jobs=njobs,
                                  decim=int(good.info['sfreq'] / 200))
        crop_pad(spec, str(pad_len)+"s")
        if epoch == "Listen":
            base = spec.copy()
        else:
            spec_a = scaling.rescale(spec, base, copy=True, mode='ratio').average(
                lambda x: np.nanmean(x, axis=0), copy=True)
            spec_a._data = np.log10(spec_a._data) * 20
            figs = chan_grid(spec_a, vlim=(-2, 20), cmap=parula_map)
            for idx, fig in enumerate(figs):
                fig.savefig(f"{fig_dir}\\{subj}_{epoch}_wavelet_{idx+1}.png")
                plt.close(fig)

        if get_band == 'gamma':
            gamma.extract(trials, copy=False, n_jobs=njobs)
            trials.resample(100)  # downsample from 2kHz to 100Hz after extracting high gamma
        elif get_band == 'theta':
            # Bandpass filter 4-8 Hz
            trials_filtered = trials.copy().filter(
                l_freq=4.0, h_freq=8.0,
                method='iir',
                iir_params={'order': 4, 'ftype': 'butter'},
                verbose=False
            )
            trials_filtered.resample(100)
            # Define frequencies - single band centered at 6 Hz
            freqs = np.array([6.0])
            n_cycles = 6
            # Compute time-frequency representation
            trials = trials_filtered.compute_tfr(
                freqs=freqs,
                method='morlet',  # Using Morlet wavelet (similar to Hilbert for single band)
                n_cycles=n_cycles,
                return_itc=False,
                average=False,
                verbose=False
            )
        # trim pad seconds on the beginning and end of the data (edge artifacts)
        crop_pad(trials, str(pad_len)+"s")
        out.append(trials)
    base = out.pop(0)

    # time perm and save stats in resampled, frequency band
    mask = dict()
    nperm = 5000
    sig2 = base.copy().get_data()
    for epoch, name, window in zip((out[0]["Audio"], out[1]["Go"], out[2]["Response"]),
            ("aud", "go", "resp"),
            ((-0.5, 1),(-0.5, 1),(-0.5, 1))):  # time-perm
        sig1 = epoch.copy().get_data(tmin=window[0], tmax=window[1])

        # time-perm
        mask[name], p_act = time_perm_cluster(
            sig1, sig2, p_thresh=0.05, axis=0, tails = 2, n_perm=nperm, n_jobs=njobs,
            ignore_adjacency=1) #tails=2 for both directions in theta band
        #power = scaling.rescale(epoch, base, 'mean', copy=True)
        z_score = scaling.rescale(epoch, base, 'zscore', copy=True)

        # build mne.EvokedArray to save the data with metadata
        if mask[name].ndim == 2:
            mask_sliced = mask[name]
            p_sliced = p_act
            z_sliced = z_score.get_data()
            base_sliced = base.get_data()
        elif mask[name].ndim == 3:
            sliced = mask[name][:, 0, :]
            p_sliced = p_act[:, 0, :]
            z_sliced = z_score.get_data()[:,:,0,:]
            base_sliced = base.get_data()[:,:,0,:]
        epoch_mask = EvokedArray(mask_sliced, epoch.average().info,
                                     tmin=window[0])
        zscore_EA = EpochsArray(z_sliced, info=z_score.info.copy(), tmin=z_score.times[0], events=z_score.events, event_id=z_score.event_id)
        # Calculate the p-value
        p_vals = EvokedArray(p_sliced, epoch_mask.info, tmin=window[0])
        # saving epoch as fif file
        #power.save(save_dir + f"/{subj}_{name}_power-epo.fif", overwrite=True, fmt='double')
        zscore_EA.save(save_dir + f"/{subj}_{name}_zscore-epo.fif", overwrite=True)
        epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.fif", overwrite=True)
        p_vals.save(save_dir + f"/{subj}_{name}_pval-ave.fif", overwrite=True)
    base_EA = EpochsArray(base_sliced, info=base.info.copy(), tmin=base.times[0],
                            events=base.events, event_id=base.event_id)
    base_EA.save(save_dir + f"/{subj}_base-epo.fif", overwrite=True)


#%% Prep data to labeledarray
from ieeg.calc.mat import LabeledArray, combine
from analysis.decoding import Decoder
from analysis.utils.mat_load import load_dict
from analysis.check.chan_utils import nested_dict_to_ndarray
import numpy as np

suffix = "zscore-epo.fif"
conds = {'aud': (-0.5,1), 'go': (-0.5,1), 'resp': (-0.5,1)}

zscores = load_dict(layout, conds, 'zscore', False, 'stgrerefgammastats')
mask = load_dict(layout, conds, 'significance', True, 'stgrerefgammastats')
zscores = combine(zscores, (0, 3)) # combine subj with channel
zscoresArray, zscoresLabel = nested_dict_to_ndarray(zscores)
zscoresLA = LabeledArray(zscoresArray, labels=zscoresLabel)
mask = combine(mask, (0, 2)) # combine subj with channel as stims already averaged
# maskArray, maskLabel = nested_dict_to_ndarray(mask)
# maskLA = LabeledArray(maskArray, labels=maskLabel)
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

#%% Get list of subjects
def get_unique_subjects(channel_labels):
    from analysis.check.chan_utils import parse_channel_info
    """
    Extract unique subject IDs from channel labels using parse_channel_info

    Parameters:
    -----------
    channel_labels : list
        List of channel labels in format 'D##-channelname'

    Returns:
    --------
    list
        Sorted list of unique subject IDs without leading zeros
    """
    # Extract subject IDs using parse_channel_info
    subject_ids = []
    for label in channel_labels:
        subid, _, _ = parse_channel_info(label)
        if subid:  # Only add if subid exists
            # Remove 'D' and leading zeros
            clean_id = f"D{int(subid[1:])}"
            subject_ids.append(clean_id)

    # Convert to set to get unique values and back to sorted list
    unique_subjects = sorted(set(subject_ids))

    return unique_subjects
# Example usage:
# channel_list = ['D017-LTM2', 'D017-LTM3', 'D050-RTM1']
# unique_subjs = get_unique_subjects(channel_list)
# print(unique_subjs)  # ['D17', 'D50']

hipp_subj = get_unique_subjects(zscoresLA.labels[1])

#%% print HG traces
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from analysis.check.chan_utils import parse_channel_info

def get_subject_id(channel_str):
    subid, _, _ = parse_channel_info(channel_str)
    return subid

def plot_HG_traces(data_array, sub_channel_list, legend=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    conditions = ['aud', 'go', 'resp']
    titles = ['Audio', 'Go', 'Response']
    subject_ids = [get_subject_id(sub_channel_list[idx]) for idx in range(sub_channel_list.shape[0])]
    unique_subjects = sorted(set(subject_ids))
    colormap = cm.get_cmap('tab20', len(unique_subjects))
    subject_to_color = {subj: colormap(i) for i, subj in enumerate(unique_subjects)}
    time = np.linspace(-0.5, 1.0, 150)  # 150 time points from -0.5 to 1.0 sec

    for idx, (cond, title) in enumerate(zip(conditions, titles)):
        for i in range(sub_channel_list.shape[0]):
            subj_id = get_subject_id(sub_channel_list[i])
            color = subject_to_color.get(subj_id, 'black')

            original = data_array[cond][i].__array__()  # shape: (208, 150)
            avg_trace = np.nanmean(original, axis=0)
            axes[idx].plot(time,avg_trace, color=color, alpha=1, linewidth=0.8)

            # Legend with subject IDs
        if legend:
            handles = [
                axes[idx].Line2D([0], [0], color=color, lw=2, label=subj)
                for subj, color in subject_to_color.items()
            ]
            if idx == 2:
                axes[idx].legend(
                    handles=handles,
                    title="Subject",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=6,
                    fontsize=9,
                    title_fontsize=10,
                    frameon=False
                )
        # Set axis limits and ticks
        axes[idx].set_ylim(-1, 4.5)
        axes[idx].set_yticks([-1, 0, 1, 2,3,4])
        axes[idx].set_xlim(-0.5, 1)
        axes[idx].set_xticks([-0.5, 0.0, 0.5, 1.0])
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)

        # Set title and axis labels
        axes[idx].set_title(title, fontsize=14)
        axes[idx].set_xlabel("Time from Onset (s)", fontsize=12)
        axes[idx].set_ylabel("Z(HG)", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(analysisfolder, 'phonemeseq_stg_reref_HG.png'), dpi=300,
                bbox_inches='tight')
    plt.show()

plot_HG_traces(zscoresLA, zscoresLA.labels[1])


# %% Time sliding decoding for reconstruction
import pickle
from analysis.decoding import classes_from_labels
from analysis.check.chan_utils import remove_min_nan_ch, equal_valid_trials_ch, left_adjust_by_stim

window_len=20 #200ms windowing
window_kwargs = {'obs_axs': 1, 'normalize': 'true', 'n_jobs': 10,
                'average_repetitions': False, 'oversample': False}
true_cat_1st = {'abae':1, 'abi':1, 'aka':1, 'aku':1, 'ava':1, 'avae':1,
                 'aeba':2, 'aebi':2, 'aebu':2, 'aega':2, 'aeka':2, 'aepi':2,
                 'ibu':3, 'ika':3, 'ikae':3, 'ipu':3, 'iva':3, 'ivu':3,
                 'uba':4, 'uga':4, 'ugae':4, 'ukae':4, 'upi':4, 'upu':4, 'uvae':4, 'uvi':4,
                 'bab':5, 'baek':5, 'bak':5, 'bup':5,
                 'gab':6, 'gaeb':6, 'gaev':6, 'gak':6, 'gav':6, 'gig':6, 'gip':6, 'gub':6,
                 'kab':7, 'kaeg':7, 'kub':7, 'kug':7,
                 'paek':8, 'paep':8, 'paev':8, 'puk':8, 'pup':8,
                 'vaeg':9, 'vaek':9, 'vip':9, 'vug':9, 'vuk':9}
true_cat_2nd = {'bab':1, 'bak':1, 'gab':1, 'kab':1,  'gak':1, 'gav':1,
                'baek':2, 'gaeb':2, 'gaev':2, 'kaeg':2, 'paek':2, 'paep':2, 'paev':2,  'vaeg':2, 'vaek':2,
                'gig':3, 'gip':3, 'vip':3,
                'bup':4, 'gub':4,'kub':4, 'kug':4, 'puk':4, 'pup':4, 'vug':4, 'vuk':4,
                'abae':5, 'abi':5, 'aeba':5, 'aebi':5, 'aebu':5, 'ibu':5, 'uba':5,
                'aega': 6, 'uga':6, 'ugae':6,
                'aka':7, 'aku':7, 'aeka':7, 'ika':7, 'ikae':7,  'ukae':7,
                'aepi':8, 'ipu':8, 'upi':8, 'upu':8,
                'ava': 9, 'avae': 9, 'iva':9, 'ivu':9,'uvae':9, 'uvi':9}
true_cat_3rd = {'aka':1, 'ava':1, 'aeba':1, 'aega':1, 'aeka':1,  'ika':1, 'iva':1, 'uba':1, 'uga':1,
                'avae':2, 'abae':2, 'ikae':2, 'ugae':2, 'ukae':2, 'uvae':2,
                'abi':3, 'aebi':3, 'aepi':3, 'upi':3, 'uvi':3,
                'aku':4, 'aebu':4, 'ibu':4, 'ipu':4, 'ivu':4,'upu':4,
                'bab':5,  'gab':5, 'gaeb':5, 'gub':5, 'kab':5, 'kub':5,
                'kaeg':6,'kug':6, 'gig':6, 'vaeg':6, 'vug':6,
                'gak':7, 'baek':7, 'bak':7, 'paek':7, 'vaek':7,  'vuk':7, 'puk':7,
                'paep':8,'pup':8, 'gip':8, 'bup':8,'vip':8,
                'gav':9, 'gaev':9,'paev':9}
true_cat_vcv = {'abae':1, 'abi':1, 'aka':1, 'aku':1, 'ava':1, 'avae':1,
                 'aeba':1, 'aebi':1, 'aebu':1, 'aega':1, 'aeka':1, 'aepi':1,
                 'ibu':1, 'ika':1, 'ikae':1, 'ipu':1, 'iva':1, 'ivu':1,
                 'uba':1, 'uga':1, 'ugae':1, 'ukae':1, 'upi':1, 'upu':1, 'uvae':1, 'uvi':1,
                 'bab':2, 'baek':2, 'bak':2, 'bup':2,
                 'gab':2, 'gaeb':2, 'gaev':2, 'gak':2, 'gav':2, 'gig':2, 'gip':2, 'gub':2,
                 'kab':2, 'kaeg':2, 'kub':2, 'kug':2,
                 'paek':2, 'paep':2, 'paev':2, 'puk':2, 'pup':2,
                 'vaeg':2, 'vaek':2, 'vip':2, 'vug':2, 'vuk':2}
#decoder_cat = {'b':5, 'g':6, 'k':7, 'p':8}
#decoder_cat = {'a':1, 'ae':2, 'i':3, 'u':4, 'b':5, 'g':6, 'k':7, 'p':8, 'v':9}
decoder_cat = {'vcv':1, 'cvc':2}
iter_num = 50
trial_num = 80
true_scores_dict = {}
for i_cond, cond in enumerate(conds.keys()):
    scores_out = np.zeros((iter_num, zscoresLA.shape[-1] - window_len + 1, len(decoder_cat), len(decoder_cat)),
                          dtype=np.float16)
    zscoresLA_cond = zscoresLA.take(i_cond, axis=0)
    decoder = Decoder(decoder_cat, 0.8, 'lda', n_splits=5, n_repeats=10)
    cats, labels = classes_from_labels(zscoresLA.labels[2], crop=slice(0, 4)) #this get out repetitions of same stims
    flipped_cats = {v:k for k,v in cats.items()} #{0: abae, 1: abi, 2: aeba, 3: aebi, 4: aebu, 5: aega}
    new_labels = np.array([true_cat_vcv[flipped_cats[l]] for l in labels]) #convert to true categories
    zscoresLA_cond_idx = zscoresLA_cond.take(sig_idx_union, axis=0)
    zscoresLA_cond_idx, _ = remove_min_nan_ch(zscoresLA_cond_idx, new_labels, min_non_nan=trial_num)
    for i_iter in range(iter_num):
        zscoresLA_cond_idx_reduced = equal_valid_trials_ch(zscoresLA_cond_idx, new_labels, min_non_nan=trial_num, upper_limit=trial_num)
        zscores_cropped, labels_cropped = left_adjust_by_stim(zscoresLA_cond_idx_reduced, new_labels, crop=True)
        valid_labels = set(decoder_cat.values())
        valid_idx = np.where(labels_cropped[np.isin(labels_cropped, list(valid_labels))])[0]
        labels_cropped = labels_cropped[valid_idx]
        zscores_cropped = zscores_cropped.take(valid_idx, axis=1)
        scores_iter = decoder.cv_cm(zscores_cropped.__array__(), labels_cropped,
                                            **window_kwargs, window=window_len)
        scores_out[i_iter] = np.mean(scores_iter, axis=1) #this averages over CV within decoder
    true_scores_dict[cond] = scores_out

with open(f'{analysisfolder}\\true_scores_phonemeseq_2way_lingreref_gamma.pkl', 'wb') as f:
    pickle.dump(true_scores_dict, f)

#%% shuffle
shuffle_scores_dict = {}
for i_cond, cond in enumerate(conds.keys()):
    scores_out = np.zeros((iter_num, zscoresLA.shape[-1] - window_len + 1, len(decoder_cat), len(decoder_cat)),
                          dtype=np.float16)
    zscoresLA_cond = zscoresLA.take(i_cond, axis=0)
    decoder = Decoder(decoder_cat, 0.8, 'lda', n_splits=5, n_repeats=10)
    cats, labels = classes_from_labels(zscoresLA.labels[2], crop=slice(0, 4)) #this get out repetitions of same stims
    flipped_cats = {v:k for k,v in cats.items()}
    new_labels = np.array([true_cat_vcv[flipped_cats[l]] for l in labels]) #convert to true categories
    zscoresLA_cond_idx = zscoresLA_cond.take(sig_idx_union, axis=0)
    zscoresLA_cond_idx, _ = remove_min_nan_ch(zscoresLA_cond_idx, new_labels, min_non_nan=trial_num)
    for i_iter in range(iter_num):
        zscoresLA_cond_idx_reduced = equal_valid_trials_ch(zscoresLA_cond_idx, new_labels, min_non_nan=trial_num, upper_limit=trial_num)
        zscores_cropped, labels_cropped = left_adjust_by_stim(zscoresLA_cond_idx_reduced, new_labels, crop=True)
        valid_idx = np.where(labels_cropped[np.isin(labels_cropped, list(valid_labels))])[0]
        labels_cropped = labels_cropped[valid_idx]
        zscores_cropped = zscores_cropped.take(valid_idx, axis=1)
        scores_iter = decoder.cv_cm(zscores_cropped.__array__(), labels_cropped,
                                            **window_kwargs, window=window_len, shuffle=True)
        scores_out[i_iter] = np.mean(scores_iter, axis=1) #this averages over CV i.e. shuffles in this case within decoder
    shuffle_scores_dict[cond] = scores_out

with open(f'{analysisfolder}\\shuffle_scores_phonemeseq_2way_lingreref_gamma.pkl', 'wb') as f:
    pickle.dump(shuffle_scores_dict, f)

#%% Approx bipolar referencing
# from analysis.check.chan_utils import parse_channel_info, build_shank_map, find_nearest_nonhipp
#
# def bipolar_reference(sub_channel_list, maxhipp_ch_label, idx_maxhipp, data_array):
#     result = np.copy(data_array)
#     shank_map = build_shank_map(sub_channel_list)
#
#     for idx in idx_maxhipp:
#         ref_idx = find_nearest_nonhipp(idx, sub_channel_list, maxhipp_ch_label, shank_map)
#         if ref_idx is not None:
#             result[idx] = data_array[idx] - data_array[ref_idx]
#         else:
#             print(f"No reference found for index {idx}: {sub_channel_list[idx]}")
#     return result
#
# zscoresLA_bipolar = zscoresLA.copy()
# for i_cond, cond in enumerate(conds.keys()):
#     zscoresLA_cond = zscoresLA.take(i_cond, axis=0)
#     bipolar_data_cond = bipolar_reference(sub_channel_list, maxhipp_ch_label, idx_maxhipp, zscoresLA_cond.__array__())
#     zscoresLA_bipolar.__array__()[i_cond] = bipolar_data_cond
#
#
# #%% compare bipolar rereferencing traces
# def plot_bipolar_comparison(data_array, bipolar_data, idx_maxhipp, sub_channel_list, num_to_plot=10, seed=42):
#     plt.figure(figsize=(15, 3 * num_to_plot))
#     if seed is not None:
#         np.random.seed(seed)
#     # Randomly select indices from idx_maxhipp without replacement
#     indices_to_plot = np.random.choice(idx_maxhipp, size=num_to_plot, replace=False)
#
#     for i, idx in enumerate(indices_to_plot):
#         original = data_array[idx]      # shape: (208, 150)
#         bipolar = bipolar_data[idx]     # shape: (208, 150)
#         original_avg = np.nanmean(original, axis=0)
#         bipolar_avg = np.nanmean(bipolar, axis=0)
#         print(f"Electrode {idx}: mean={original_avg.mean():.4f}, std={original_avg.std():.4f}")
#
#         plt.subplot(num_to_plot, 1, i + 1)
#         plt.plot(original_avg, label="Original", color='gray')
#         plt.plot(bipolar_avg, label="Bipolar", color='blue')
#         plt.title(f"Electrode: {sub_channel_list[idx]} (Index {idx})")
#         plt.xlabel("Time Points")
#         plt.ylabel("Signal")
#         plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
# plot_bipolar_comparison(zscoresLA_cond.__array__(), bipolar_data_aud, idx_hipp, sub_channel_list, num_to_plot=10)

#%%
# with open(os.path.join(analysisfolder, 'true_scores_phonemeseq_9way_1stphoneme_bipolar.pkl'), 'rb') as f:
#     true_scores_dict = pickle.load(f)

mean_scores_dict = {key: np.mean(scores, axis=0) for key, scores in true_scores_dict.items()}
mean_scores_dict = {key: np.mean(scores[40:90, :, :], axis=0) for key, scores in mean_scores_dict.items()}
def plot_confusion_matrices(scores_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    conditions = ['aud', 'go', 'resp']
    titles = ['Audio', 'Go', 'Response']

    # Define labels for the axes
    labels = ['VCV', 'CVC']

    for idx, (cond, title) in enumerate(zip(conditions, titles)):
        # Get mean over iterations and time points 40-90
        mean_cm = scores_dict[cond]

        # Plot confusion matrix
        im = axes[idx].imshow(mean_cm, cmap='RdBu_r', vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=12)

        # Add labels
        axes[idx].set_xticks(np.arange(len(labels)))
        axes[idx].set_yticks(np.arange(len(labels)))
        axes[idx].set_xticklabels(labels)
        axes[idx].set_yticklabels(labels)

        # Rotate x labels
        plt.setp(axes[idx].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values in cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = axes[idx].text(j, i, f'{mean_cm[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(analysisfolder, 'confusion_matrices_2way_stgreref_firsthalfsec_gamma.png'), dpi=300,
                bbox_inches='tight')
    plt.show()


# Call the function with your true_scores_dict
plot_confusion_matrices(mean_scores_dict)


#%% old plot code for single pickle file
from ieeg.calc.stats import time_perm_cluster
def plot_mean_with_std(data, timepoints, ax, color, label):
    mean_trace = np.mean(data, axis=0)
    std_trace = np.std(data, axis=0)
    ax.plot(timepoints, mean_trace, label=label, color=color, linewidth=1.5)
    ax.fill_between(timepoints, mean_trace - std_trace, mean_trace + std_trace,
                    color=color, alpha=0.2, linewidth=0)

with open(os.path.join(analysisfolder, 'true_scores_phonemeseq_2way_m1reref_gamma.pkl'), 'rb') as f:
    true_scores_dict = pickle.load(f)
with open(os.path.join(analysisfolder, 'shuffle_scores_phonemeseq_2way_m1reref_gamma.pkl'), 'rb') as f:
    shuffle_scores_dict = pickle.load(f)

timepoints = np.linspace(-0.4, 0.9, 131)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
bar_width = 0.01  # LDA run every 10ms intervals with 200ms window
y_position = 0.8
xlim = (-0.4, 0.9)
ylim = (0.4,1)
for ax in fig.axes:
    ax.axhline(0.5, color='k', linestyle='--')
for i, cond in enumerate(true_scores_dict.keys()):
    true_traces = np.mean(np.diagonal(true_scores_dict[cond], axis1=2, axis2=3), axis=2)  # traces of average decoding accuracy
    shuffle_traces = np.mean(np.diagonal(shuffle_scores_dict[cond], axis1=2, axis2=3), axis=2)
    signif = time_perm_cluster(true_traces, shuffle_traces, 0.05, n_perm = 5000, stat_func=lambda x, y, axis: np.mean(x, axis=axis))

    plot_mean_with_std(true_traces, timepoints, ax = axes[i], color = 'red', label = 'true')
    plot_mean_with_std(shuffle_traces, timepoints, ax = axes[i], color = 'blue', label = 'shuffle')

    # Set labels and title for each subplot
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Decoding Score')
    axes[i].set_title(f'Score Trace for {cond}')
    axes[i].set_xlim(xlim)
    axes[i].set_ylim(ylim)
    axes[i].legend()
    x_axis = np.append(np.arange(xlim[0], xlim[1], bar_width), xlim[1])
    for idx, bool_value in enumerate(signif[0]):
        if bool_value:
            axes[i].barh(y=y_position, width=bar_width, height=0.01, left=x_axis[idx], color='black')
plt.suptitle('Non-M1(mouth/larynx) re-referencing')
plt.tight_layout()
plt.savefig(os.path.join(analysisfolder, 'phonemeseq_2way_m1reref_gamma.png'), dpi=300,
                bbox_inches='tight')
plt.show()
