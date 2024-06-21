import torch
from sca import SCA, get_sample_weights
from analysis.grouping import GroupData
from ieeg.viz.ensemble import subgrids
from ieeg.calc.mat import LabeledArray
import os
import numpy as np
from sklearn import config_context
import scipy.stats as st
import matplotlib.pyplot as plt

## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# Create a gridspec instance with 3 rows and 3 columns


## Load the data
kwarg_sets = [dict(folder='stats'), dict(folder='stats', wide=True)]
fnames = ["short", "wide"]
groups = ['AUD', 'SM', 'PROD', 'sig_chans']
colors = ['green', 'red', 'blue', 'grey']
for i, (kwargs, fname) in enumerate(zip(kwarg_sets, fnames)):
    sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs)
    break

aud_slice = slice(0, 175)
pval = np.where(sub.p_vals > 0.9999, 0.9999, sub.p_vals)

zscores = LabeledArray(st.norm.ppf(1 - pval), sub.p_vals.labels)
powers = np.nanmean(sub['zscore'].array, axis=(-4, -2))

trainp = np.hstack([zscores['aud_ls', :, aud_slice],
                    zscores['aud_lm', :, aud_slice],
                    zscores['aud_jl', :, aud_slice],
                    zscores['go_ls'],
                    zscores['go_lm'],
                    zscores['go_jl'],
                    zscores['resp']])

conds_aud = ['aud_ls', 'aud_lm', 'aud_jl']
conds_go = ['go_ls', 'go_lm', 'go_jl']
aud = sub.array['zscore', conds_aud]
aud.labels[0] = aud.labels[0].replace("aud_", "")
go = sub.array['zscore', conds_go]
go.labels[0] = go.labels[0].replace("go_", "")
aud_go = aud[..., :175].concatenate(go, -1)

zs = np.nanmean(aud_go, axis=(1, 3))

idx = list(sub.SM)
X = trainp[idx].T - np.min(trainp[idx])
n = 7
sca = SCA(n_components=n,
          init='rand',
          lam_sparse=0.1)
          # orth=True)
weights = get_sample_weights(X)
sca.fit(X, sample_weight=weights)
Y = sca.reconstruct(X).T
r2 = sca.r2_score
# plt.plot(sca.losses)
_, axs = plt.subplots(3, 1, sharex=True,sharey=True, gridspec_kw={'hspace': 0})
pows = np.array([sca.transform(zs[j, idx].T).T for j in range(3)])
for i in range(n):
    for j in range(len(axs)):
        axs[j].plot(pows[j, i])
        axs[j].axvline(50, color='black', linestyle='--')
        axs[j].axvline(175, color='black')
        axs[j].axvline(225, color='black', linestyle='--')
        axs[j].ylabel = zs.labels[0][j]


# %% plot the dot product of the learned U weights (which are not constrained to be orthonormal)
U = sca.params['U']
U_dProd = U.T@U

# plot
# fig2 = plt.imshow(U_dProd)