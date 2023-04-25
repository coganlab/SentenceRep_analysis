import mne
import os
import numpy as np
from ieeg.io import get_data
from ieeg.viz.utils import plot_dist, plot_clustering
from ieeg.calc.utils import stitch_mats
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
epochs, all_power, names = load_intermediates(layout, conds, "zscore")
signif, all_sig, _ = load_intermediates(layout, conds, "significance")

# %% plot significant channels
AUD, SM, PROD, sig_chans = group_elecs(all_sig, names, conds)

# %% plot groups
aud_c = "aud_ls"
go_c = "go_ls"
stitch_aud = stitch_mats([all_power[aud_c][AUD, :150],
                         all_power[go_c][AUD, :]], [0], axis=1)
stitch_sm = stitch_mats([all_power[aud_c][SM, :150],
                        all_power[go_c][SM, :]], [0], axis=1)
stitch_prod = stitch_mats([all_power[aud_c][PROD, :150],
                          all_power[go_c][PROD, :]], [0], axis=1)
stitch_all = np.vstack([stitch_aud, stitch_sm, stitch_prod])
labels = np.concatenate([np.ones([len(AUD)]), np.ones([len(SM)]) * 2,
                        np.ones([len(PROD)]) * 3])
plot_clustering(stitch_all, labels, sig_titles=['AUD', 'SM', 'PROD'],
                colors=[[0,1,0],[1,0,0],[0,0,1]])


# cond = 'go_jl'
# plot_dist(all_power[cond][AUD], times=conds[cond], label='AUD', color='g')
# plot_dist(all_power[cond][SM], times=conds[cond], label='SM', color='r')
# plot_dist(all_power[cond][PROD], times=conds[cond], label='PROD', color='b')
# plt.legend()
# plt.xlabel("Time(s)")
# plt.ylabel("High Gamma Power (V)")
# plt.title("Go")
