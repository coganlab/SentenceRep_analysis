## Preprocess
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq import gamma, utils
from ieeg.calc import stats, scaling
import os.path as op
import os
import mne
from itertools import product


n_jobs = -1
## check if currently running a slurm job
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
    if int(subj[1:]) in (3, 65, 71):
        continue
    # Load the data
    TASK = "SentenceRep"
    # subj = "D" + str(sub).zfill(4)
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension='.edf', desc='clean', preload=False)

    if "Trigger" in filt.ch_names:
        filt.drop_channels(["Trigger"])

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

    for epoch, t in zip(("Start", "Start", "Word"),
                        ((-0.5, 0), (-0.5, 0.5), (-1, 1.5))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True
                            , reject_by_annotation=False)
        outliers_to_nan(trials, outliers=10)
        gamma.extract(trials, copy=False, n_jobs=n_jobs)
        utils.crop_pad(trials, "0.5s")
        trials.resample(100)
        trials.filenames = good.filenames
        out.append(trials)
        # if len(out) == 2:
        #     break

    base = out.pop(0)

    # power = scaling.rescale(out[1], out[0], copy=True, mode='mean')
    # power.average(method=lambda x: np.nanmean(x, axis=0)).plot()
    ## run time cluster stats

    save_dir = op.join(layout.root, "derivatives", "stats")
    if not op.isdir(save_dir):
        os.mkdir(save_dir)
    mask = dict()
    data = []
    nperm = 10000
    sig2 = base.get_data(copy=True)
    for epoch, name, window in zip((out[0]["Start"],) +
            tuple(out[1][e] for e in ["Response"] + list(
                map("/".join, product(["Audio", "Go"], ["LS", "LM", "JL"])))),
            ("start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "go_jl"),
            ((-0.5, 0.5), (-1, 1), *((-0.5, 1.5),) * 6)):  # time-perm
            # (*((0, 0.5),) * 5, *((0.25, 0.75),) * 3)):  # ave
        sig1 = epoch.get_data(tmin=window[0], tmax=window[1], copy=True)

        # time-perm
        mask[name], p_act = stats.time_perm_cluster(
            sig1, sig2, p_thresh=0.05, axis=0, n_perm=nperm, n_jobs=n_jobs,
            ignore_adjacency=1)
        epoch_mask = mne.EvokedArray(mask[name], epoch.average().info,
                                     tmin=window[0])

        # # ave
        # mask[name] = stats.window_averaged_shuffle(sig1, sig2, 10000)
        # epoch_mask = mne.EvokedArray(mask[name][:, None], epoch.average().info,
        #                              tmin=window[0])
        # p_vals = epoch_mask.copy()

        power = scaling.rescale(epoch, base, 'mean', copy=True)
        z_score = scaling.rescale(epoch, base, 'zscore', copy=True)
        # sig2 = stats.make_data_same(sig2, sig1.shape)

        # Calculate the difference between the two groups averaged across
        # out = st.permutation_test([sig1, sig2], stats.mean_diff,
        #                           n_resamples=nperm,
        #                           vectorized=True, alternative='less',
        #                           batch=500)
        # p_act = out.pvalue

        # Calculate the p-value
        p_vals = mne.EvokedArray(p_act, epoch_mask.info, tmin=window[0])
        # p_vals = epoch_mask.copy()
        data.append((name, epoch_mask.copy(), power.copy(), z_score.copy(), p_vals.copy()))

    for name, epoch_mask, power, z_score, p_vals in data:
        power.save(save_dir + f"/{subj}_{name}_power-epo.fif", overwrite=True,
                   fmt='double')
        z_score.save(save_dir + f"/{subj}_{name}_zscore-epo.fif", overwrite=True,
                     fmt='double')
        epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.fif", overwrite=True)
        p_vals.save(save_dir + f"/{subj}_{name}_pval-ave.fif", overwrite=True)

    base.save(save_dir + f"/{subj}_base-epo.fif", overwrite=True)
    del data

    ## Plot
    # import matplotlib.pyplot as plt  # noqa E402
    # plt.imshow(mask['go_ls'])#[:, None])
