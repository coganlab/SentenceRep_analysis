from analysis.grouping import GroupData
from ieeg.viz.ensemble import plot_dist, subgrids
import os
import numpy as np
import scipy.stats as st

## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# Create a gridspec instance with 3 rows and 3 columns
r = 4
c_minor = 3
c_major = 3
major_rows = (0, 1)

fig, axs = subgrids(r, c_major, c_minor, major_rows)


## Load the data
kwargs = dict(folder='stats')
fnames = ["All", "GM_a2009s", "GM_BN_atlas"]
groups = ['AUD', 'SM', 'PROD', 'sig_chans']
colors = ['green', 'red', 'blue', 'grey']
wm = [None, ".a2009s", ".BN_atlas"]
sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs)
for i, fname in enumerate(fnames):

    idxs = [list(getattr(sub, group)) for group in groups]
    if wm[i] is not None:
        sub.atlas = wm[i]
        idxs = [list(set(idx) & set(sub.grey_matter)) for idx in idxs]

    subfig = sub.plot_groups_on_average(idxs[:-1], hemi='lh', colors=colors[:-1])
    screenshot = subfig.screenshot()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    axs[0][i].imshow(cropped_screenshot)
    axs[0][i].set_title(fname)
    axs[0][i].axis('off')

    # %% more plots
    idx_count = [len(idx) for idx in idxs]

    if i == 0:
        axs[1][i].table(cellText=np.array([[idx_count]]).T, rowLabels=groups,
                        loc='center')
    else:
        axs[1][i].table(cellText=np.array([[idx_count]]).T, loc='center')
    axs[1][i].axis('off')

    # %% plot the distribution sub.power in aud_ls, go_ls, and resp as subplots
    conds = {'aud_ls': (-0.5, 1.5), 'go_ls': (-0.5, 1.5), 'resp': (-1, 1)}

    # make a plot for each condition in conds as a subgrid
    for j, cond in enumerate(conds):
        arr = np.nanmean(sub.array['zscore', cond].__array__(), axis=(0, 2))
        ax = axs[2][i][j]
        for group, color, idx in zip(groups[:-1], colors[:-1], idxs[:-1]):
            plot_dist(arr[idx], times=conds[cond],
                      label=group, ax=ax, color=color)
        if j == 0:
            # ax.legend()
            ax.set_ylabel("Z-Score (V)")
            ylims = ax.get_ylim()
        # ax.set_xlabel("Time(s)")
        ax.set_ylim(ylims)
        ax.set_title(cond)

    # %% plot the distribution sub.power in aud_ls, go_ls, and resp as subplots
    conds = {'aud_ls': (-0.5, 1.5), 'go_ls': (-0.5, 1.5), 'resp': (-1, 1)}

    # make a plot for each condition in conds as a subgrid
    for j, cond in enumerate(conds):
        pval = sub.p_vals[cond]
        pval = np.where(pval > 0.9999, 0.9999, pval)
        arr = st.norm.ppf(1 - pval)
        if fname == 'ave':
            arr = np.tile(arr, (1, 200))

        ax = axs[3][i][j]
        for group, color, idx in zip(groups[:-1], colors[:-1], idxs[:-1]):
            plot_dist(arr[idx], times=conds[cond],
                      label=group, ax=ax, color=color)
        if j == 0:
            if i == 0:
                ax.legend()
                ax.set_ylabel("Z-Score (pvals)")
            ylims = ax.get_ylim()
        ax.set_xlabel("Time(s)")
        ax.set_ylim(ylims)
        ax.set_title(cond)


## setup training data
aud_slice = slice(0, 175)
stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                      # sub.signif['aud_lm', :, aud_slice],
                      sub.signif['resp', :]])
                      # sub.signif['resp', :]])

pval = np.hstack([sub.p_vals['aud_ls', :, aud_slice],
                      # sub.signif['aud_lm', :, aud_slice],
                      sub.p_vals['resp', :]])
                      # sub.signif['resp', :]])
pval = np.where(pval > 0.9999, 0.9999, pval)

# pval[pval<0.0001] = 0.0001
zscores = st.norm.ppf(pval)
powers = np.nanmean(sub['zscore'].array, axis=(-4, -2))
sig = sub.signif

# trainz = np.hstack([zscores['aud_ls', :, aud_slice],
#                    # zscores['aud_lm', :, aud_slice],
#                    zscores['resp']])
#                     # zscores['resp']])
trainp = np.hstack([powers['aud_ls', :, aud_slice],
                   # powers['aud_lm', :, aud_slice],
                   powers['resp']])

# %%
# r2 = []
# with torch.cuda.device('cuda:0'):
#     X = torch.from_numpy(trainp - np.min(trainp))
#     X = X[list(sub.AUD)].T
#     with config_context(array_api_dispatch=True):
#         model = SCA(orth=True, lam_sparse=0.01, init='rand')
#         for i in range(2, 8):
#             print(f"n_components: {i}/7")
#             model.n_components = i
#             W = model.fit_transform(X).T
#             Y = model.reconstruct(X).T
#             r2.append(model.r2_score)
#
# plt.plot(r2)