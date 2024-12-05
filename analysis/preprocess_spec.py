from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
from ieeg.calc import stats, scaling
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from joblib import Parallel, delayed
import mne
import os
import os.path as op
from itertools import product
from tqdm import tqdm
import numpy as np


n_jobs = 11

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    sid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    subjects = [f"D{sid:04}"]
    layout = get_data("SentenceRep", root=LAB_root)
    print(f"Running subject {subjects[0]}")
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
for subj in subjects:
    if int(subj[1:]) not in (29,):
        continue
    # Load the data
    TASK = "SentenceRep"
    # %%
    save_dir = op.join(layout.root, "derivatives", "stats_freq")
    if not op.isdir(save_dir):
        os.mkdir(save_dir)
    mask = dict()
    data = []
    nperm = 10000
    spec_type = 'hilbert'
    filename = os.path.join(layout.root, 'derivatives',
                            'spec', spec_type, subj, f'start-tfr.h5')
    base = mne.time_frequency.read_tfrs(filename).crop(-0.5, 0)
    sig2 = base.get_data()
    for name, window in zip(
            ("start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "go_jl"),
            ((-0.5, 0.5), (-1, 1), *((-0.5, 1.5),) * 6)):  # time-perm
            # (*((0, 0.5),) * 5, *((0.25, 0.75),) * 3)):  # ave

        filename = os.path.join(layout.root, 'derivatives',
                                    'spec', spec_type, subj, f'{name}-tfr.h5')
        epoch = mne.time_frequency.read_tfrs(filename)
        sig1 = epoch.get_data(tmin=window[0], tmax=window[1])

        # time-perm
        mask[name], p_act = stats.time_perm_cluster(
            sig1, sig2, p_thresh=0.05, axis=0, n_perm=nperm, n_jobs=n_jobs,
            ignore_adjacency=1)
        epoch_mask = mne.time_frequency.AverageTFRArray(
            epoch.average().info, mask[name], epoch.times, epoch.freqs, nave=1)

        # Plot the Time-Frequency Clusters
        # --------------------------------
        # figs = chan_grid(epoch_mask, size=(20, 10), vmin=0, vmax=1,
        #                  cmap=parula_map, show=False)

        power = scaling.rescale(epoch, base, 'mean', copy=True)
        z_score = scaling.rescale(epoch, base, 'zscore', copy=True)

        # Calculate the p-value
        p_vals = mne.time_frequency.AverageTFRArray(epoch_mask.info, p_act,
                                                    epoch.times, epoch.freqs)
        # p_vals = epoch_mask.copy()
        data.append((name, epoch_mask.copy(), power.copy(), z_score.copy(),
                     p_vals.copy()))

    for name, epoch_mask, power, z_score, p_vals in data:
        power.save(save_dir + f"/{subj}_{name}_power-epo.h5", overwrite=True,
                   fmt='double')
        z_score.save(save_dir + f"/{subj}_{name}_zscore-epo.h5", overwrite=True,
                     fmt='double')
        epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.h5", overwrite=True)
        p_vals.save(save_dir + f"/{subj}_{name}_pval-ave.h5", overwrite=True)

    base.save(save_dir + f"/{subj}_base-epo.h5", overwrite=True)
