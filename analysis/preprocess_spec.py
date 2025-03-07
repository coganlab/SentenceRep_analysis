from ieeg.io import get_data
from ieeg.calc import stats, scaling
import mne
import os
import os.path as op


# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    sid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    subject = 71

n_jobs = -1

for sub in subjects:
    if int(sub[1:]) in (32, 30):
        continue
    # Load the data
    TASK = "SentenceRep"
    # %%
    save_dir = op.join(layout.root, "derivatives", "stats_freq_multitaper")
    if not op.isdir(save_dir):
        os.mkdir(save_dir)
    mask = dict()
    data = []
    nperm = 5000
    spec_type = 'multitaper'
    filename = os.path.join(layout.root, 'derivatives',
                            'spec', spec_type, sub, f'start-tfr.h5')
    base = mne.time_frequency.read_tfrs(filename).crop(-0.5, 0., 50, 500)
    sig2 = base.get_data()
    for name, window in zip(
            ("start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "go_jl"),
            ((-0.5, 0.5), (-1, 1), *((-0.5, 1.5),) * 6)):  # time-perm
            # (*((0, 0.5),) * 5, *((0.25, 0.75),) * 3)):  # ave

        # if os.path.exists(save_dir + f"/{sub}_{name}_mask-tfr.h5"):
        #     continue

        filename = os.path.join(layout.root, 'derivatives',
                                    'spec', spec_type, sub, f'{name}-tfr.h5')
        epoch = mne.time_frequency.read_tfrs(filename)
        sig1, times, freqs = epoch.get_data(tmin=window[0], tmax=window[1],
                                            fmin=50, fmax=500,
                              return_times=True, return_freqs=True)

        # time-perm
        mask[name], p_act = stats.time_perm_cluster(
            sig1, sig2, p_thresh=0.05, axis=0, n_perm=nperm, n_jobs=n_jobs,
            ignore_adjacency=1)
        epoch_mask = mne.time_frequency.AverageTFRArray(
            epoch.average().info, mask[name], times, freqs, nave=1)

        # Plot the Time-Frequency Clusters
        # --------------------------------
        # figs = chan_grid(epoch_mask, size=(20, 10), vmin=0, vmax=1,
        #                  cmap=parula_map, show=False)

        power = scaling.rescale(epoch.crop(fmin=50, fmax=500), base, 'mean', copy=True)
        z_score = scaling.rescale(epoch.crop(fmin=50, fmax=500), base, 'zscore', copy=True)

        # Calculate the p-value
        p_vals = mne.time_frequency.AverageTFRArray(epoch_mask.info, p_act,
                                                    times, freqs)
        p_vals = epoch_mask.copy()
        data.append((name, epoch_mask.copy(), power.copy(), z_score.copy(),
                     p_vals.copy()))

        # for name, epoch_mask, power, z_score, p_vals in data:
        power.save(save_dir + f"/{sub}_{name}_power-tfr.h5", overwrite=True)
        z_score.save(save_dir + f"/{sub}_{name}_zscore-tfr.h5", overwrite=True)
        epoch_mask.save(save_dir + f"/{sub}_{name}_mask-tfr.h5", overwrite=True)
        p_vals.save(save_dir + f"/{sub}_{name}_pval-tfr.h5", overwrite=True)

    base.save(save_dir + f"/{sub}_base-tfr.h5", overwrite=True)
