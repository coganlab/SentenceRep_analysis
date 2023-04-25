import mne
import os
import numpy as np
from ieeg.io import get_data
from ieeg.viz import plot_dist
import matplotlib.pyplot as plt
from utils.mat_load import load_intermediates, group_elecs


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
epochs, all_power, names = load_intermediates(layout, conds, "power")
signif, all_sig, _ = load_intermediates(layout, conds, "significance")

# %% plot significant channels
AUD, SM, PROD, sig_chans = group_elecs(all_sig, names, conds)

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
