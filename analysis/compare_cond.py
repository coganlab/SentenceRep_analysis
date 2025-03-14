from analysis.grouping import GroupData, group_elecs
from ieeg.viz.ensemble import plot_dist, subgrids
from ieeg.io import DataLoader, get_data
from ieeg.arrays.label import LabeledArray, combine, Labels
import mne
import os
import numpy as np

def average_tfr_channels(tfr: mne.time_frequency.tfr.AverageTFR):
    info = mne.create_info(ch_names=['Average'], sfreq=tfr.info['sfreq'])
    avgd_data = np.nanmean(tfr.data, axis=0, keepdims=True)
    return mne.time_frequency.AverageTFRArray(info, avgd_data, tfr.times, tfr.freqs)

def name_from_idx(idx: list[int], chs: Labels):
    return [f"D{int(p[0][1:])}-{p[1]}" for p in
             (s.split("-") for s in chs[idx])]

## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
# Create a gridspec instance with 3 rows and 3 columns
r = 4
c_minor = 2
c_major = 2
major_rows = (0,)

fig, axs = subgrids(r, c_major, c_minor, major_rows)

## Load the data
kwargs = [dict(folder='stats'), dict(folder='stats_freq_multitaper')]
fnames = ['gamma', 'freq']
groups = ['AUD', 'SM', 'PROD', 'sig_chans']
colors = ['green', 'red', 'blue', 'grey']
# wm = [None, ".a2009s", ".BN_atlas"]

for i, fname in enumerate(fnames):
    folder = kwargs[i]['folder']
    if folder == 'stats':
        sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs[i])
        idxs = [list(getattr(sub, group)) for group in groups]
        # if wm[i] is not None:
        #     sub.atlas = wm[i]
        #     idxs = [list(set(idx) & set(sub.grey_matter)) for idx in idxs]

        subfig = sub.plot_groups_on_average(idxs[:-1], hemi='lh',
                                            colors=colors[:-1])
        zscores = np.nanmean(sub.array['zscore'], axis=(1, 3))
    else:
        conds_all = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                     "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                     "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                     "go_jl": (-0.5, 1.5)}
        loader = DataLoader(layout, conds_all, 'significance', True, folder,
                            '.h5')
        filemask = os.path.join(layout.root, 'derivatives', folder, 'combined',
                                'mask')
        if not os.path.exists(filemask + ".npy"):
            sigs = LabeledArray.from_dict(combine(loader.load_dict(
                dtype=bool, n_jobs=-1), (0, 2)), dtype=bool)
            sigs.tofile(filemask)
        else:
            sigs = LabeledArray.fromfile(filemask)

        AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1],
                                               sigs.labels[0])
        idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans}
        idxs = [sorted(idxs[group]) for group in groups]

        filez = os.path.join(layout.root, 'derivatives', folder, 'combined',
                             'zscores')
        loader = DataLoader(layout, conds_all, 'zscore', False, folder,
                            '.h5')
        if not os.path.exists(filez + ".npy"):
            zscores = LabeledArray.from_dict(combine(loader.load_dict(
                dtype="float16", n_jobs=-1), (0, 3)), dtype="float16")
            zscores.tofile(filez)
        else:
            trials = LabeledArray.fromfile(filez, mmap_mode='r')
            zscores = np.nanmean(trials, axis=(1,3,4))

        from ieeg.viz.mri import plot_on_average

        subfig = None
        for k, idx in enumerate(idxs[:-1]):
            picks = name_from_idx(idx, zscores.labels[1])
            subfig = plot_on_average(layout.get_subjects(), picks=picks, hemi='both',
                                 color=colors[k], fig=subfig)

    screenshot = subfig.screenshot()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    axs[0][i].imshow(cropped_screenshot)
    axs[0][i].set_title(fname)
    axs[0][i].axis('off')

    # %% plot the distribution sub.power in aud_ls, go_ls, and resp as subplots
    for k, condss in enumerate(
            [['aud_jl', 'go_jl'], ['aud_lm', 'go_lm'], ['aud_ls', 'go_ls']]):
        conds = {condss[0]: (-0.5, 1.5), condss[1]: (-0.5, 1.5)}

        # make a plot for each condition in conds as a subgrid
        for j, cond in enumerate(conds):
            arr = zscores[cond].__array__()
            ax = axs[k + 1][i][j]
            for group, color, idx in zip(groups[:-1], colors[:-1], idxs[:-1]):
                plot_dist(arr[idx], times=conds[cond],
                          label=group, ax=ax, color=color)
            if j == 0:
                # ax.legend()
                ax.set_ylabel("Z-Score (V), " + cond[-2:])
                ylims = ax.get_ylim()
            # ax.set_xlabel("Time(s)")
            ax.set_ylim(ylims)
            if k == 0:
                if cond.split("_")[0] == "aud":
                    ax.set_title("Stimulus")
                else:
                    ax.set_title("Go Cue")

            elif k == 2:
                ax.set_xlabel("Time(s)")

