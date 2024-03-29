from analysis.grouping import GroupData
from analysis.utils.plotting import plot_weight_dist
import os
import numpy as np
import scipy.stats as st
from sca import SCA
import matplotlib.pyplot as plt
import torch
from sklearn import config_context
import pyvista as pv

## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
fig, axs = plt.subplots(4, 3)

## Load the data
sub_ave = GroupData.from_intermediates("SentenceRep", fpath,
                                   folder='ave', fdr=True)
subfig = sub_ave.plot_groups_on_average(rm_wm=False, hemi='lh')
axs[0, 0].imshow(subfig.screenshot())

sub = GroupData.from_intermediates("SentenceRep", fpath,
                                   folder='stats', wide=False)

subfig = sub.plot_groups_on_average(rm_wm=False, hemi='lh')
axs[0, 1].imshow(subfig.screenshot())
sub_wide = GroupData.from_intermediates("SentenceRep", fpath,
                                        folder='stats', wide=True)
subfig = sub_wide.plot_groups_on_average(rm_wm=False, hemi='lh')
axs[0, 2].imshow(subfig.screenshot())


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