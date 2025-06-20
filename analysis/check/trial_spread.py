from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, crop_empty_data, outliers_to_nan, find_bad_channels_lof
from ieeg.calc import stats, scaling
from ieeg.timefreq import gamma
import os
from ieeg.timefreq.utils import crop_pad, resample_tfr
import ieeg.viz
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import functools

## check if currently running a slurm job
HOME = os.path.expanduser("~")

if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    subject = None

n_jobs = 10
fig, axs = plt.subplots(5, 4, figsize=(20, 10))
ylims = [0, 0]
n = 0
for i, sub in enumerate(s for s in subjects if int(s[1:]) not in (30, 32, 71)):
    if subject is not None:
        if int(sub[1:]) != subject:
            continue

    if i - n >= 20:
        fig, axs = plt.subplots(5, 4, figsize=(20, 10))
        n += 20
    # Load the data
    filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                           extension='.edf', desc='notch', preload=False)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt,).copy()

    # good.info['bads'] = channel_outlier_marker(good, 3, 2)
    bads = good.info['bads']
    good.info['bads'] = []
    # good.info['bads'] = channel_outlier_marker(good, 3, 2)
    # bads = list(set(filt.info['bads']) | set(find_bad_channels_lof(good, n_jobs=n_jobs)))
    good.drop_channels(bads)
    good.load_data()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    # scheme = pre.make_contact_rereference_arr(good.ch_names)
    # good._data = pre.rereference(scheme, field=[good._data.T])[0].T
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'super_log', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    trials = {}
    # fig, axs = plt.subplots(5, 7, figsize=(20, 10))

    for epoch, t, name in zip(
            ("Start", "Word/Response/LS",),
            ((-0.5, 0), (-1, 1),),
            ("start", "resp",)):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials[name] = trial_ieeg(good, epoch, times, preload=True)
        # outliers_to_nan(trials[name], outliers=10, tmin=t[0], tmax=t[1])
        func = functools.partial(st.iqr, rng=(50, 95), nan_policy='omit')
        outliers_to_nan(trials[name], outliers=6, deviation=func,
                        center=np.nanmedian, tmin=t[0], tmax=t[1])
        gamma.extract(trials[name], copy=False, n_jobs=n_jobs)
        crop_pad(trials[name], "0.5s")
        outliers_to_nan(trials[name], outliers=6, deviation=func,
                        center=np.nanmedian)
        # outliers_to_nan(trials[name], outliers=12)
        if name == "start":
            # base = trials[name].copy().crop(-0.5, 0)
            continue

        scaling.rescale(trials[name], trials["start"], 'mean', copy=False)
        #

        isnan = np.isnan(trials[name].get_data()).any(axis=-1).T
        maxmax = np.max(trials[name].get_data(), axis=-1).T
        # maxmax = reduce(lambda x, y: np.concatenate((x, y), axis=-1),
        #                 (a for a in trials[name].get_data()))
        # isnan = np.isnan(maxmax)
        title = f" {name}: {np.sum(isnan, dtype=float) / isnan.size * 100:.1f}% NaN"

        # axs[0, i].imshow(np.isnan(trials[name].get_data()).any(axis=-1),
        #                  aspect='auto', interpolation='nearest')
        # axs[0, i].set_title(title)
        j, k = divmod(i - n, 4)
        ax = axs[j, k]
        ax.boxplot([d[~m] for d, m in zip(maxmax, isnan)], notch=True, vert=True)
        ax.set_xticks(range(0, len(trials[name].ch_names), 25))
        ax.set_xticklabels(range(0, len(trials[name].ch_names), 25))
        if j == 0:
            ax.set_ylabel("Trials")
        # else:
            # axs[0, i].set_yticklabels([])
            # ax.set_yticklabels([])
        # y = ax.get_ylim()
        # ylims[0] = min(ylims[0], y[0])
        # ylims[1] = max(ylims[1], y[1])


    # for ax in axs:
    # ax.set_ylim(ylims)
    ax.set_title(sub + title)
# fig.suptitle(sub)
fig.supxlabel("Channel")
fig.tight_layout()

