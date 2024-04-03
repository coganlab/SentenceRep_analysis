import os
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

from analysis.grouping import GroupData
from ieeg.viz.ensemble import plot_weight_dist
import sklearn.decomposition as skd
from scipy.sparse import csr_matrix, issparse, linalg as splinalg

import matplotlib.pyplot as plt

## Load the data
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath,
                                   folder='stats_opt', wide=False)
## setup training data
aud_slice = slice(0, 175)
stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                      # sub.signif['aud_lm', :, aud_slice],
                      sub.signif['resp', :]])
pval = np.hstack([sub.p_vals['aud_ls', :, aud_slice],
                  # sub.signif['aud_lm', :, aud_slice],
                  sub.p_vals['resp', :]]) ** 4
# sub.signif['resp', :]])

zscores = np.nanmean(sub['zscore'].array, axis=(-4, -2))
powers = np.nanmean(sub['power'].array, axis=(-4, -2))
sig = sub.signif

trainz = np.hstack([zscores['aud_ls', :, aud_slice],
                    # zscores['aud_lm', :, aud_slice],
                    zscores['resp']])
# zscores['resp']])
trainp = np.hstack([powers['aud_ls', :, aud_slice],
                    # powers['aud_lm', :, aud_slice],
                    powers['resp']])
# powers['resp']])
# raw = train - np.min(train)
sparse_matrix = csr_matrix((trainz[stitched == 1], stitched.nonzero()))
sparse_matrix.data -= np.min(sparse_matrix.data)


##
options = dict(init="random", max_iter=100000, solver='cd', shuffle=False,
               tol=1e-13, l1_ratio=1)
idx = list(sub.SM & sub.grey_matter)
model = skd.NMF(**options)
# idx = list(sub.SM & sub.grey_matter)
W, H, n = skd.non_negative_factorization(sparse_matrix[idx],
                                         n_components=3,
                                         **options)
# W = H.T
# W *= np.mean(zscores) / np.mean(powers) / 1000
# this_plot = np.hstack([sub['aud_ls'].sig[aud_slice], sub['go_ls'].sig])

##
conds = {"resp": (-1, 1),
         "aud_ls": (-0.5, 1.5),
         "aud_lm": (-0.5, 1.5),
         "aud_jl": (-0.5, 1.5),
         "go_ls": (-0.5, 1.5),
         "go_lm": (-0.5, 1.5),
         "go_jl": (-0.5, 1.5)}
metric = zscores
labeled = [['Instructional', 'c', 2],
           ['Motor', 'm', 0],
           ['Feedback', 'k', 1]]
# ['Working Memory', 'orange', 3]]
# ['Auditory', 'g', 4]]
pred = np.argmax(W, axis=1)
groups = [[idx[i] for i in np.where(pred == j)[0]]
          for j in range(W.shape[1])]
colors, labels = zip(
    *((c[1], c[0]) for c in sorted(labeled, key=lambda x: x[2])))
for i, g in enumerate(groups):
    labeled[i][2] = groups[labeled[i][2]]
cond = 'aud_ls'
if cond.startswith("aud"):
    title = "Stimulus Onset"
    times = slice(None)
    ticks = [-0.5, 0, 0.5, 1, 1.5]
    legend = True
elif cond.startswith("go"):
    title = "Go Cue"
    times = slice(None)
    ticks = [-0.5, 0, 0.5, 1, 1.5]
    legend = False
else:
    title = "Response"
    times = slice(None)
    ticks = [-1, -0.5, 0, 0.5, 1]
    legend = False
plot = metric[cond, idx, times].__array__().copy()
# plot[sig[cond, sub.SM, times].__array__() == 0] = np.nan
fig, ax = plot_weight_dist(plot, pred, times=conds[cond], colors=colors,
                           sig_titles=labels)
plt.xticks(ticks)
plt.rcParams.update({'font.size': 14})
# plt.ylim(0, 1.5)
# sort by order
labeled = sorted(labeled, key=lambda x: np.median(
    np.argmax(metric[cond, x[2]].__array__(), axis=1)))
ylim = ax.get_ylim()

for i, (label, color, group) in enumerate(labeled):
    points = metric[cond, group, times].__array__()
    # points[sig[cond, group, times].__array__() == 0] = 0
    peaks = np.max(points, axis=1)
    peak_locs = np.argmax(points, axis=1)
    tscale = np.linspace(conds[cond][0], conds[cond][1], metric.shape[-1])
    # ax.scatter(tscale[peak_locs], peaks, color=colors[i], label=labels[i])
    # plot horizontal box plot of peak locations
    spread = (ylim[1] - ylim[0])
    width = spread / 16
    pos = [width / 2 + i * width + spread * 3 / 4]
    # boxplot_2d(tscale[peak_locs], peaks, ax)
    bplot = ax.boxplot(tscale[peak_locs], manage_ticks=False, widths=width,
                       positions=pos, boxprops=dict(facecolor=color,
                                                    fill=True, alpha=0.5),
                       vert=False, showfliers=False, whis=[15, 85],
                       patch_artist=True, medianprops=dict(color='k'))

trans = ax.get_xaxis_transform()
# ax.text(50, 0.8, title, rotation=270, transform=trans)
plt.ylabel("Z-scored HG power")
plt.xlabel(f"Time from {title} (s)")
plt.xlim(*conds[cond])
if legend:
    plt.legend()
plt.tight_layout()
# plt.axhline(linestyle='--', color='k')
# plt.axvline(linestyle='--', color='k')
# plt.savefig(cond+'_peaks.svg', dpi=300)

## run the decomposition
# ch = [[] for _ in range(9)]
# scorer = lambda model: ch[model.fit.rank-1].append(calinski_halbaraz(
#     np.array(model.fit.W).T, np.array(model.fit.H)))
# options = dict(seed="nndsvd", rank=4, max_iter=10000,
#                # callback=scorer,
#                update='divergence',
#                objective='div',
#                options=dict(flag=0)
#                )
# nmf = nimfa.Bmf(stitched[sub.SM], **options)
# nmf.factorize()
# plot_weight_dist(stitched[sub.SM, 175:550], np.array(nmf.W))
##
# est = nmf.estimate_rank(list(range(1, 6)))
# matplotlib.pyplot.plot([e['evar'] / e['rss'] for e in est.values()])
# ##
# W, H = nmf.fitted()
# # W = np.array(bmf.W)
# this_plot = np.hstack([sub[:,'aud_ls'].signif[aud_slice], sub[:,'go_ls'].signif])

## plot on brain
pred = np.argmax(W, axis=1)
groups = [[sub.keys['channel'][idx[i]] for i in np.where(pred == j)[0]]
          for j in range(W.shape[1])]
fig1 = sub.plot_groups_on_average(groups,
                                  ['k', 'm', 'c', 'red'],
                                  hemi='lh',
                                  rm_wm=False)

### plot the weights
# fig, ax = plt.subplots(1, 1)
# ax.set_ylim([np.min(W), np.max(W)])

