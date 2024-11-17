import os
import scipy.io
import numpy as np
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
    ch_label = labels[ch]
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
        sub_num, ch = sep_sub_ch(sub_ch)
        info = mri.subject_to_info(f"D{sub_num}", sub_dir)
        labels = mri.find_labels(info, sub=f"D{sub_num}", subj_dir=sub_dir,
                    atlas=".BN_atlas", hot_words=hot_words, pct_thresh=pct_thresh)
        channel_label.append(labels[ch])
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

