from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
from ieeg.calc import stats
from ieeg.viz.utils import chan_grid
from ieeg.viz.parula import parula_map
from joblib import Parallel, delayed
import os
from itertools import product
from tqdm import tqdm
import numpy as np


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
    labels = (out[0]["Start"],) + tuple(out[1][e] for e in ["Response"] + list(
                map("/".join, product(["Audio", "Go"], ["LS", "LM", "JL"]))))
    names = ("start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm",
             "go_jl"),
    for epoch, name, t in tqdm(zip(
            labels, names, ((-0.5, 0.5), (-1, 1), *((-0.5, 1.5),) * 6)),
            total=len(labels)):
        # if name != 'resp':
        #     continue
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        outliers_to_nan(epoch, outliers=10)
        spec = wavelet_scaleogram(epoch, n_jobs=n_jobs, decim=int(
            good.info['sfreq'] / 100))
        crop_pad(spec, "0.5s")
        base_fixed = stats.make_data_same(base._data, spec._data.shape)
        mask = spec.average(lambda x: np.nanmean(x, axis=0), copy=True)

        temp = np.mean(stats.shuffle_test(spec._data, base_fixed, 1000), axis=0)
        mask._data = temp
        # mask = stats.time_perm_cluster(spec._data, base._data,
        #                                p_thresh=0.05,
        #                                ignore_adjacency=1,
        #                                n_jobs=n_jobs,
        #                                # ignore channel adjacency
        #                                n_perm=2000)
        out2.append(spec)
        masks.append(mask)

        # Plot the Time-Frequency Clusters
        # --------------------------------
        figs = chan_grid(mask, size=(20, 10), vmin=0, vmax=1,
                         cmap=parula_map, show=False)
