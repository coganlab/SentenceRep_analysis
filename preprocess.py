## Preprocess
import ieeg.viz.utils
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq import gamma, utils
from ieeg.calc import stats, scaling
from ieeg.process import parallelize
import numpy as np
import os.path as op
import os
import mne

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
    if 17 < int(subj[1:]) or int(subj[1:]) < 9:
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

    for epoch, t in zip(("Start", "Word/Response/LS", "Word/Audio/LS",
                         "Word/Audio/LM", "Word/Audio/JL", "Word/Speak/LS",
                         "Word/Mime/LM", "Word/Audio/JL"),
                        ((-0.5 , 0), (-1, 1), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5),
                         (-0.5, 1.5), (-0.5, 1.5), (1, 3))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True#, reject_tmin=t[0],
                            # reject_tmax=t[1]
                            , reject_by_annotation=False)
        outliers_to_nan(trials, outliers=10)
        gamma.extract(trials, copy=False, n_jobs=1)
        utils.crop_pad(trials, "0.5s")
        trials.resample(100)
        trials.filenames = good.filenames
        out.append(trials)
        # if len(out) == 2:
        #     break

    base = out.pop(0)


    ## run time cluster stats

    save_dir = op.join(layout.root, "derivatives", "stats")
    if not op.isdir(save_dir):
        os.mkdir(save_dir)
    mask = dict()
    for epoch, name in zip(out, ("resp", "aud_ls", "aud_lm", "aud_jl", "go_ls",
                                     "go_lm", "go_jl")):
        sig1 = epoch.get_data()
        sig2 = base.get_data()
        mask[name] = stats.time_perm_cluster(sig1, sig2, p_thresh=0.05, axis=0,
                                             n_perm=1000, n_jobs=4, ignore_adjacency=1)
        epoch_mask = mne.EvokedArray(mask[name], epoch.average().info)
        power = scaling.rescale(epoch, base, copy=True)
        power.save(save_dir + f"/{subj}_{name}_power-epo.fif", overwrite=True,
                   fmt='double')
        z_score = scaling.rescale(epoch, base, 'zscore', copy=True)
        z_score.save(save_dir + f"/{subj}_{name}_zscore-epo.fif", overwrite=True,
                     fmt='double')
        epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.fif", overwrite=True)

    ## Plot
    import matplotlib.pyplot as plt  # noqa E402
    plt.imshow(mask['go_ls'])
