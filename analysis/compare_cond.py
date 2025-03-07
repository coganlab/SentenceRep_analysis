from analysis.grouping import GroupData
from ieeg.viz.ensemble import plot_dist, subgrids
import os
import numpy as np
import scipy.stats as st

## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# Create a gridspec instance with 3 rows and 3 columns
r = 4
c_minor = 2
c_major = 1
major_rows = (0,)

fig, axs = subgrids(r, c_major, c_minor, major_rows)

## Load the data
kwargs = [dict(folder='stats')]
fnames = ["new"]
groups = ['AUD', 'SM', 'PROD', 'sig_chans']
colors = ['green', 'red', 'blue', 'grey']
# wm = [None, ".a2009s", ".BN_atlas"]

for i, fname in enumerate(fnames):
    sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs[i])
    idxs = [list(getattr(sub, group)) for group in groups]
    # if wm[i] is not None:
    #     sub.atlas = wm[i]
    #     idxs = [list(set(idx) & set(sub.grey_matter)) for idx in idxs]

    subfig = sub.plot_groups_on_average(idxs[:-1], hemi='lh',
                                        colors=colors[:-1])
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
            arr = np.nanmean(sub.array['zscore', cond].__array__(),
                             axis=(0, 2))
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

