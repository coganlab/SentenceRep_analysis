from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
from ieeg.calc import stats, scaling
import os
from itertools import product
n_jobs = 6

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
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension='.edf', desc='clean', preload=False)

    # %%
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

    # %%
    out = []
    for epoch, t in zip(("Start", "Start", "Word"),
                        ((-0.5, 0), (-0.5, 0.5), (-1, 1.5))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True
                            , reject_by_annotation=False)
        trials.filenames = good.filenames
        out.append(trials)
        # if len(out) == 2:
        #     break
    base = wavelet_scaleogram(out.pop(0), n_jobs=n_jobs, decim=int(
            good.info['sfreq'] / 100))
    crop_pad(base, "0.5s")
    # power = scaling.rescale(out[1], out[0], copy=True, mode='mean')
    # power.average(method=lambda x: np.nanmean(x, axis=0)).plot()

    # %%
    masks = []
    out2 = []
    for epoch, name, t in zip((out[0]["Start"],) +
                              tuple(out[1][e] for e in ["Response"] + list(
                                  map("/".join, product(["Audio", "Go"],
                                                        ["LS", "LM", "JL"])))),
                              ("start", "resp", "aud_ls", "aud_lm", "aud_jl",
                               "go_ls", "go_lm", "go_jl"),
                              ((-0.5, 0.5), (-1, 1),
                               *((-0.5, 1.5),) * 6)):  # time-perm
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        outliers_to_nan(epoch, outliers=10)
        spec = wavelet_scaleogram(epoch, n_jobs=n_jobs, decim=int(
            good.info['sfreq'] / 100))
        crop_pad(spec, "0.5s")
        mask = stats.time_perm_cluster(spec._data, base._data,
                                       p_thresh=0.05,
                                       ignore_adjacency=1,
                                       n_jobs=n_jobs,
                                       # ignore channel adjacency
                                       n_perm=2000)
        out2.append(spec)
        masks.append(mask)

