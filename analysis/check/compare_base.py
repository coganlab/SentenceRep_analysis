## Preprocess
import ieeg.viz.utils
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq import gamma, utils
from ieeg.calc import stats, scaling, reshape
import os.path as op
import os
import mne
from itertools import product
import numpy as np


## check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")

for subj in subjects:
    if int(subj[1:]) in (3, 32, 65, 71):
        continue
    # Load the data
    TASK = "SentenceRep"
    # subj = "D" + str(sub).zfill(4)
    layout = get_data("SentenceRep", root=LAB_root)
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension='.edf', desc='clean', preload=False)

    ## Crop raw data to minimize processing time
    new = crop_empty_data(filt, )

    # Mark channel outliers as bad
    # new.info['bads'] = channel_outlier_marker(new, 4)
    # Exclude bad channels
    good = new.copy().drop_channels(filt.info['bads'])
    good.load_data()

    # CAR
    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    # Remove intermediates from mem
    del new

    ## High Gamma Filter and epoching
    out = []

    for epoch, t in zip(("Start", "Word"),
                        ((-0.5, 0), (-1, 1.5))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True
                            , reject_by_annotation=False)
        outliers_to_nan(trials, outliers=10)
        gamma.extract(trials, copy=False, n_jobs=-3)
        utils.crop_pad(trials, "0.5s")
        trials.resample(100)
        trials.filenames = good.filenames
        out.append(trials)
        # if len(out) == 2:
        #     break

    base = out.pop(0)
    ## Compare baseline with make_data_same and regular in a histogram
    data = base.get_data(copy=True)
    for epoch, name, window in zip(
            (out[0][e] for e in ["Response"] + list(
                map("/".join, product(["Audio", "Go"], ["LS", "LM", "JL"])))),
            ("resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "go_jl"),
            ((-1, 1), *((-0.5, 1.5),) * 6)):  # time-perm
            # ((-0.25, 0.25), *((0, 0.5),) * 3, *((0.25, 0.75),) * 3)):  # ave
        sig1 = epoch.get_data(tmin=window[0], tmax=window[1], copy=True)
        break
    data_re = reshape.make_data_same(data, sig1.shape)


    ## plot histogram
    import matplotlib.pyplot as plt
    cols = 5
    rows = int(np.ceil(data.shape[1] / cols))
    fig1, ax1 = plt.subplots(rows, cols)
    fig2, ax2 = plt.subplots(rows, cols)
    for i in range(data.shape[1]):
        b_mean, b_std = stats.dist(data[:, i], -1, 'std')
        r_mean, r_std = stats.dist(data_re[:, i], -1, 'std')
        r, c = divmod(i, cols)
        ax1[r, c].hist(b_mean, bins=50, alpha=0.5, label="base")
        ax1[r, c].hist(r_mean, bins=50, alpha=0.5, label="re")
        ax1[r, c].set_title(epoch.ch_names[i])
        ax2[r, c].hist(b_std, bins=50, alpha=0.5, label="base")
        ax2[r, c].hist(r_std, bins=50, alpha=0.5, label="re")
        ax2[r, c].set_title(epoch.ch_names[i])
    plt.legend()
