import os
import scipy.io
import numpy as np
from ieeg.calc.mat import LabeledArray
from ieeg.io import get_data, raw_from_layout

def sep_sub_ch(sub_ch:str):
    """
    Input: subject_channel string
    Output: separates into subject number and channel
    """
    sub = sub_ch.split('-')[0]
    sub_num = sub.lstrip('D0') # this strips all leading 0s
    ch = sub_ch.split('-')[1]
    return sub_num, ch

def regen_ch_label(sub_ch: str, sub_dir: str):
    """
    Input: full subj_channel name
    Output: ROI label using Brainnectome atlas with radius 10mm
    """
    from ieeg.viz import mri
    sub_num, ch = sep_sub_ch(sub_ch)
    info = mri.subject_to_info(f"D{sub_num}", sub_dir)
    labels = mri.gen_labels(info, sub=f"D{sub_num}", subj_dir=sub_dir,
                    atlas=".BN_atlas")
    try:
        ch_label = labels[ch]
    except KeyError:
        ch_label = 'ERROR'
        print(f'Error reading {sub_ch}')
    return ch_label

def get_ch_label(channels, sub_dir):
    """
    Input: list of channels
    Output: list of atlas labels
    """
    channel_label = []
    for sub_ch in channels:
        channel_label.append(regen_ch_label(sub_ch, sub_dir))
    return channel_label

def get_preferred_ch_label(channels, sub_dir, hot_words, pct_thresh):
    from ieeg.viz import mri
    channel_label = []
    for sub_ch in channels:
        try:
            sub_num, ch = sep_sub_ch(sub_ch)
            info = mri.subject_to_info(f"D{sub_num}", sub_dir)
            labels = mri.find_labels(info, sub=f"D{sub_num}", subj_dir=sub_dir,
                        atlas=".BN_atlas", hot_words=hot_words, pct_thresh=pct_thresh)
            channel_label.append(labels[ch])
        except KeyError:
            channel_label.append('ERROR')
            print(f'Error reading {sub_ch}')
    return channel_label

def get_muscle_chans(matdir: str, matfname: str, subj: str):
    import scipy.io
    try:
        subj = 'D' + str(int(subj[1:]))  # removing leading 0 to match folder name
        matpath = os.path.join(matdir, subj, matfname)
        if os.path.isfile(matpath):
            muscle_chans = scipy.io.loadmat(matpath)
            if len(muscle_chans['muscleChannel']) > 0:
                flattened = [chan[0] for chan in muscle_chans['muscleChannel'] if
                             chan[0].size > 0]  # this gives list of subj-channel
                muscle_channels = [chan[0].split('-')[1] for chan in flattened if
                                       '-' in chan[0]]  # produce set of only channel names
                print(f'{subj} muscle channels: {muscle_channels}')
                return muscle_channels
            else:
                print(f'{subj} muscle channels: NONE')
                return list()
        else:
            print(f'{subj}: muscleChannel file not present')
            return list()
    except OSError as e:
        print(f"Error reading {subj} mat file: {e}")
        return list()

def remove_min_nan_ch(X: LabeledArray, stim_labels: np.ndarray, min_non_nan: int = 3):
    """
    remove channels with fewer than min_nan valid trials in any category in stim_labels
    """
    nonnan_trials = np.array(~np.any(np.isnan(X.__array__()), axis=2)) #ch * trial collapsed over timepoints
    stim_types = len(set(stim_labels))
    good_ch = []
    for ch in range(X.shape[0]):
        unique_values, counts = np.unique(stim_labels[nonnan_trials[ch,:]], return_counts=True)
        # if either missing categories, or trials within a category is fewer than min_nan, discard channel
        if len(unique_values) < stim_types or np.any(counts < min_non_nan):
            continue
        good_ch.append(ch)
    return X.take(good_ch, axis = 0), good_ch

def equal_valid_trials_ch(X: LabeledArray, stim_labels: np.ndarray, min_non_nan: int=3, upper_limit: int = None):
    """
    randomly sub-select equal count of valid trials per category per channel, with conditional of min trials per category >= min_nan
    i.e. undersampling and fill in NaNs for additional valid trials.
    If upper_limit is provided, sets the maximum number of valid trials per category per channel.
    NB. X_out is fed to cv_cm and then sample_fold where mixup is run. I.e. this limits the VALID data fed into LDA but does NOT
    balance the trials by category (which is determined in stim_labels)
    """
    X_out = X.copy()
    nonnan_trials = np.array(~np.any(np.isnan(X.__array__()), axis=2)) #ch * trial collapsing over timepoints
    stim_count = len(set(stim_labels))
    stim_set = set(stim_labels)
    for ch in range(X.shape[0]):
        nonnan_labels_ch = stim_labels[nonnan_trials[ch,:]]
        unique_values, counts = np.unique(nonnan_labels_ch, return_counts=True)
        # if either missing categories, or trials within a category is fewer than min_nan, raise Error
        if len(unique_values) < stim_count or np.any(counts < min_non_nan):
            raise ValueError('input LabeledArray with channels missing categories or with trials<min_non_nan, run remove_min_nan_ch first')
        min_count = np.min(counts) # get the least number of valid trials in all categories to undersample to
        if upper_limit is not None:
            min_count = upper_limit
        for s in stim_set:
            stim_in_ch = np.array(np.where(stim_labels == s))
            valid_trials = stim_in_ch[nonnan_trials[ch, stim_in_ch]] # mask out invalid trials
            if len(valid_trials)-min_count == 0: #no need to further drop trials
                continue
            trials_to_nan = np.random.choice(valid_trials, len(valid_trials)-min_count, replace=False)
            X_out[ch, trials_to_nan, :] = np.nan
            #print(f'for ch:{ch}, stim:{s}, got valid trials:{len(valid_trials)}, undersampled: {len(trials_to_nan)}')
    return X_out

def left_adjust_by_stim(X: LabeledArray, stim_labels: np.ndarray, crop: bool = False):
    X_out = X.copy()
    labels_out = stim_labels.copy()
    nonnan_trials = np.array(~np.any(np.isnan(X.__array__()), axis=2)) # channel * trial boolean array
    stim_set = np.unique(stim_labels) # preserve order just in case
    nonnan_count_ch = np.zeros((X.shape[0], len(stim_set))).astype(int) # ch * token valid trials count
    for ch in range(X.shape[0]):
        for i,s in enumerate(stim_set):
            nonnan_count_ch[ch,i] = np.sum(nonnan_trials[ch, stim_labels == s])
    if crop:
        for s in range(len(stim_set)):
            if len(np.unique(nonnan_count_ch[:,s])) > 1:
                raise ValueError('Error with cropping: make sure nonnan_count per class is consistent across channels')
        X_out = X_out.take(range(sum(nonnan_count_ch[0,:])), axis = 1) # crop X_out size to the right trial dimension
        labels_out = np.zeros(sum(nonnan_count_ch[0,:]))
    for i,s in enumerate(stim_set):
        stim_in_ch = np.array(np.where(stim_labels == s)[0])
        for ch in range(X.shape[0]):
            nonnan_data = X[ch, stim_in_ch[nonnan_trials[ch, stim_in_ch]],:]
            if not crop:
                X_out[ch, stim_in_ch[:nonnan_count_ch[ch,i]],:] = nonnan_data # put valid data on the left
                X_out[ch, stim_in_ch[nonnan_count_ch[ch,i]:],:] = np.nan # nan out remaining trials
            else:
                smin = 0 + sum(nonnan_count_ch[ch,0:i])
                smax = smin + nonnan_count_ch[ch,i]
                X_out[ch,smin:smax,:] = nonnan_data
                labels_out[smin:smax] = s
    return X_out, labels_out

if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        LAB_root = os.path.join(HOME, "workspace", "CoganLab")
        subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:  # if not then set box directory
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("Phoneme_sequencing", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")

    #%%
    matdir = os.path.join(LAB_root, 'D_Data', 'Phoneme_Sequencing')
    matfname = 'muscleChannelWavelet.mat'
    for subj in subjects:
        if int(subj[1:]) == 0:
            continue
        else:
            get_muscle_chans(matdir, matfname, subj)

