## Preprocess
import ieeg.viz.utils
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg, channel_outlier_marker
from ieeg.timefreq import gamma, utils
from ieeg.calc import stats, scaling
import os.path as op
import os
import mne
from itertools import product


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
        gamma.extract(trials, copy=False, n_jobs=-1)
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
    for epoch, name, window in zip(
            (out[0][e] for e in ["Response"] + list(
                map("/".join, product(["Audio", "Go"], ["LS", "LM", "JL"])))),
            ("resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "go_jl"),
            ((-1, 1), *((-0.5, 1.5),) * 6)):  # time-perm
            # ((-0.25, 0.25), *((0, 0.5),) * 3, *((0.25, 0.75),) * 3)):  # ave
        sig1 = epoch.get_data(tmin=window[0], tmax=window[1])
        sig2 = base.get_data()

        # time-perm
        mask[name] = stats.time_perm_cluster(sig1, sig2, p_thresh=0.05, axis=0,
                                             n_perm=2000, n_jobs=-2,
                                             ignore_adjacency=1)
        epoch_mask = mne.EvokedArray(mask[name], epoch.average().info,
                                     tmax=window[1], tmin=window[0])

        # ave
        # mask[name] = stats.window_averaged_shuffle(sig1, sig2, 0.01, 2000)
        # epoch_mask = mne.EvokedArray(mask[name][:, None], epoch.average().info)

        power = scaling.rescale(epoch, base, 'mean', copy=True)
        z_score = scaling.rescale(epoch, base, 'zscore', copy=True)
        data.append((name, epoch_mask.copy(), power.copy(), z_score.copy()))

    for name, epoch_mask, power, z_score in data:
        power.save(save_dir + f"/{subj}_{name}_power-epo.fif", overwrite=True,
                   fmt='double')
        z_score.save(save_dir + f"/{subj}_{name}_zscore-epo.fif", overwrite=True,
                     fmt='double')
        epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.fif", overwrite=True)
    del data

    ## Plot
    # import matplotlib.pyplot as plt  # noqa E402
    # plt.imshow(mask['go_ls'])#[:, None])
