import mne
import os
import numpy as np
from ieeg.io import get_data
from ieeg.viz import plot_dist
import matplotlib.pyplot as plt


# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)
conds = {"resp": (-1, 1),
         "aud_ls": (-0.5, 1.5),
         "aud_lm": (-0.5, 1.5),
         "aud_jl": (-0.5, 1.5),
         "go_ls": (-0.5, 1.5),
         "go_lm": (-0.5, 1.5),
         "go_jl": (-0.5, 1.5)}

# %% Load the data
chn_names = []
epochs = dict()
signif = dict()
all_power = dict()
all_sig = dict()
for cond in conds.keys():
    all_power[cond] = np.empty((0, 200))
    all_sig[cond] = np.empty((0, 200))
folder = os.path.join(layout.root, 'derivatives', 'stats')
for subject in layout.get_subjects():
    epochs[subject] = dict()
    signif[subject] = dict()
    for cond in conds.keys():
        try:
            epochs[subject][cond] = mne.read_epochs(os.path.join(
                folder, f"{subject}_{cond}_power-epo.fif"))
            signif[subject][cond] = mne.read_evokeds(os.path.join(
                folder, f"{subject}_{cond}_mask-ave.fif"))[0]
        except FileNotFoundError:
            continue
        power = epochs[subject][cond].average()
        montage = power.get_montage()
        names = [subject + '-' + ch for ch in power.ch_names]

        # add new channels to list if not already there
        chn_names = chn_names + [ch for ch in names if ch not in chn_names]

        # add new channels to power and significance matrix
        all_power[cond] = np.vstack((all_power[cond], power.get_data()))
        all_sig[cond] = np.vstack((all_sig[cond],
                                   signif[subject][cond].get_data()))

# %% plot significant channels
sig_chans = []
AUD = []
SM = []
PROD = []
for i, name in enumerate(chn_names):
    for cond in conds.keys():
        if np.any(all_sig[cond][i] == 1):
            sig_chans.append(i)
            break

    audls_is = np.any(all_sig['aud_ls'][i][50:175] == 1)
    audlm_is = np.any(all_sig['aud_lm'][i][50:175] == 1)
    audjl_is = np.any(all_sig['aud_jl'][i][50:175] == 1)
    mime_is = np.any(all_sig['go_lm'][i] == 1)
    speak_is = np.any(all_sig['go_ls'][i] == 1)

    if audls_is and audlm_is and mime_is and speak_is:
        SM.append(i)
    elif audls_is and audlm_is and audjl_is:
        AUD.append(i)
    elif mime_is and speak_is:
        PROD.append(i)

# %% plot groups

cond = 'go_jl'
plot_dist(all_power[cond][AUD], times=conds[cond], label='AUD', color='g')
plot_dist(all_power[cond][SM], times=conds[cond], label='SM', color='r')
plot_dist(all_power[cond][PROD], times=conds[cond], label='PROD', color='b')
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("z-score")
plt.title("Go")
plt.ylim(-2, 15)