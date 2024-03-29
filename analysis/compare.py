from analysis.grouping import GroupData
from analysis.utils.plotting import plot_dist
import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import gridspec

## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# Create a gridspec instance with 3 rows and 3 columns
r = 4
c_minor = 3
c_major = 2
major_rows = (0, 1)

c = c_minor * c_major
gs = gridspec.GridSpec(r, c)

# Adjust the space between subplots to 0
gs.update(wspace=0, hspace=0)

# Create subplots
axs = np.zeros((r, c), dtype=object)
for i in range(r):  # Only for the first two rows
    for j in range(c):  # Create 3 subplots per row
        if i in major_rows:  # which rows will show minor columns
            if j >= c_major:
                continue
            axs[i, j] = plt.subplot(gs[i, j*c_minor:(j+1)*c_minor])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        else:
            axs[i, j] = plt.subplot(gs[i, j])
            if j != 0:
                axs[i, j].set_yticks([])
            if i != r - 1:
                axs[i, j].set_xticks([])


## Load the data
kwarg_sets = [dict(folder='stats'), dict(folder='stats', wide=True)]
fnames = ["short", "wide"]
groups = ['AUD', 'SM', 'PROD', 'sig_chans']
colors = ['green', 'red', 'blue', 'grey']
for i, (kwargs, fname) in enumerate(zip(kwarg_sets, fnames)):
    sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs)
    subfig = sub.plot_groups_on_average(rm_wm=False, hemi='lh')
    axs[0, i].imshow(subfig.screenshot())
    axs[0, i].set_title(fname)

    # %% more plots
    idx_count = []
    for group in groups:
        idx_count += [len(getattr(sub, group))]

    if i == 0:
        axs[1, i].table(cellText=np.array([[idx_count]]).T, rowLabels=groups,
                        loc='center')
    else:
        axs[1, i].table(cellText=np.array([[idx_count]]).T, loc='center')
    # %% plot the distribution sub.power in aud_ls, go_ls, and resp as subplots
    conds = {'aud_ls': (-0.5, 1.5), 'go_ls': (-0.5, 1.5), 'resp': (-1, 1)}

    # make a plot for each condition in conds as a subgrid
    for j, cond in enumerate(conds):
        arr = np.nanmean(sub.array['zscore', cond].__array__(), axis=(0, 2))
        ax = axs[2, i*c_minor+j]
        for group, color in zip(groups[:-1], colors[:-1]):
            plot_dist(arr[list(getattr(sub, group))], times=conds[cond],
                      label=group, ax=ax, color=color, mode='std')
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

        ax = axs[3, i*c_minor+j]
        for group, color in zip(groups[:-1], colors[:-1]):
            plot_dist(arr[list(getattr(sub, group))], times=conds[cond],
                      label=group, ax=ax, color=color, mode='std')
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