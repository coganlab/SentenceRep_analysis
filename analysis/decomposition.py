import os
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_weight_dist, plot_dist, boxplot_2d
import nimfa
import sklearn.decomposition as skd
from sklearn.metrics import pairwise_distances
import sys
from scipy.sparse import csr_matrix, issparse

sys._excepthook = sys.excepthook
def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)
sys.excepthook = exception_hook


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from scipy.linalg import svd
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda) ** 3 - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol: break
    return dot(Phi, R)


def _check_array(arr: np.ndarray | csr_matrix) -> np.ndarray:
    if issparse(arr):
        return arr.toarray('C')
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise ValueError("Input must be a numpy array or sparse matrix")

def calinski_halbaraz(X_in: np.ndarray | csr_matrix,
                      W_in: np.ndarray | csr_matrix, n_clusters: int = None
                      ) -> float:
    """
    Here, X is the original data matrix, W and H are the non-negative factor
    matrices from the ONMF decomposition. The function first computes the
    centroid of each cluster defined by the columns of W by taking the mean
    of the data points assigned to that cluster. It then computes the total
    sum of squares (TSS), which is the sum of squared distances between each
    data point and the mean of all data points. It computes the
    between-cluster sum of squares (BSS), which is the sum of squared
    distances between each cluster centroid and the mean of all data points,
    weighted by the number of data points assigned to that cluster. Finally,
    it computes the within-cluster sum of squares (WSS), which is the sum of
    squared distances between each data point and its assigned cluster
    centroid, weighted by the assignment weights in W.
    """

    X = _check_array(X_in)
    W = _check_array(W_in)

    if n_clusters is None:
        n_clusters = W.shape[1]
    n_samples = X.shape[0]

    # Compute the centroid of each cluster
    centroids = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        centroids[i] = np.mean(X[W[:, i] > 0], axis=0)

    # Compute the between-cluster sum of squares
    BSS = np.sum(np.sum(W[:, i, np.newaxis] * pairwise_distances(
        centroids[i, np.newaxis, :], np.mean(X, axis=0, keepdims=True))**2)
                 for i in range(n_clusters))

    # Compute the within-cluster sum of squares
    WSS = np.sum(np.sum(W[:, i, np.newaxis] * pairwise_distances(
        X, centroids[i, np.newaxis, :])**2) for i in range(n_clusters))

    # Compute the CH index
    ch = (BSS / (n_clusters - 1)) / (WSS / (n_samples - n_clusters))

    return ch


def mat_err(W: np.ndarray, H: np.ndarray, orig: np.ndarray) -> float:
    error = np.linalg.norm(orig - W @ H) ** 2 / np.linalg.norm(orig) ** 2
    return error


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ## Load the data
    fpath = os.path.expanduser("~/Box/CoganLab")
    sub = GroupData.from_intermediates("SentenceRep", fpath,
                                       folder='stats_old')
    conds = {"resp": (-1, 1),
             "aud_ls": (-0.5, 1.5),
             "aud_lm": (-0.5, 1.5),
             "aud_jl": (-0.5, 1.5),
             "go_ls": (-0.5, 1.5),
             "go_lm": (-0.5, 1.5),
             "go_jl": (-0.5, 1.5)}
    ## setup training data
    aud_slice = slice(0, 175)
    stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                          sub.signif['aud_lm', :, aud_slice],
                          sub.signif['go_ls', :],
                          sub.signif['resp', :]])
    zscores = np.nanmean(sub['zscore'].array, axis=(-4, -2))
    powers = np.nanmean(sub['power'].array, axis=(-4, -2))
    train = zscores
    sig = sub.signif

    train = np.hstack([train['aud_ls', :, aud_slice],
                       train['aud_lm', :, aud_slice],
                       train['go_ls'], train['resp']])
    combined = train * stitched - np.min(train * stitched)
    raw = train - np.min(train)
    sparse_matrix = csr_matrix((combined[stitched == 1], stitched.nonzero()))

    ## try clustering
    options = dict(init="random", n_components=4, max_iter=10000, solver='mu',
                   beta_loss='kullback-leibler',
                   tol=1e-7, verbose=1)
    W, H, n = skd.non_negative_factorization(sparse_matrix[sub.SM], **options)
    # W *= np.mean(zscores) / np.mean(powers) / 1000
    # this_plot = np.hstack([sub['aud_ls'].sig[aud_slice], sub['go_ls'].sig])
    ##
    metric = zscores
    labeled = [['Instructional', 'c',1],
               ['Motor', 'm', 3],
               ['Feedback', 'k', 2],
               ['Working Memory', 'orange', 0]]
    pred = np.argmax(W, axis=1)
    groups = [[sub.SM[i] for i in np.where(pred == j)[0]]
              for j in range(W.shape[1])]
    colors, labels = zip(*((c[1], c[0]) for c in sorted(labeled, key=lambda x: x[2])))
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
    plot = metric[cond, sub.SM, times].__array__().copy()
    # plot[sig[cond, sub.SM, times].__array__() == 0] = np.nan
    fig, ax = plot_weight_dist(plot, pred, times=conds[cond], colors=colors, sig_titles=labels)
    plt.xticks(ticks)
    plt.rcParams.update({'font.size': 14})
    plt.ylim(0, 1.5)
    # sort by order
    labeled = sorted(labeled, key=lambda x: np.median(np.argmax(metric[cond, x[2]].__array__(), axis=1)))
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
        pos = [width/2+i * width + spread*3 / 4]
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
    plt.savefig(cond+'_decomp_ord.svg', dpi=300,)

    ## plot on brain
    groups = [[sub.keys['channel'][sub.SM[i]] for i in np.where(pred == j)[0]]
                for j in range(W.shape[1])]
    fig = sub.plot_groups_on_average(groups, colors, hemi='lh',
                                     # rm_wm=False,
                                     size=0.4)
    fig.save_image('SM_decomp.eps')

    ## Find and plot the peaks of each electrode group as a scatter plot / horizontal box plot
    groups_idx = [[sub.SM[i] for i in np.where(pred == j)[0]]
                for j in range(W.shape[1])]
    fig, ax = plt.subplots(1, 1)
    # ax.set_ylim([np.min(zscores[cond, sub.SM].__array__()),
    #              np.max(zscores[cond, sub.SM].__array__())])
    plt.ylim(0.25, 4)
    plt.xlim(*conds[cond])
    ylim = ax.get_ylim()
    for i, group in enumerate(groups):
        peaks = np.max(zscores[cond, group].__array__(), axis=1)
        peak_locs = np.argmax(zscores[cond, group].__array__(), axis=1)
        tscale = np.linspace(conds[cond][0], conds[cond][1], 200)
        ax.scatter(tscale[peak_locs], peaks, color=colors[i], label=labels[i])
        # plot horizontal box plot of peak locations
        spread = (ylim[1] - ylim[0])
        width = spread / 16
        pos = [spread / 2 + i * width + 1.5 * width]
        bplot = ax.boxplot(tscale[peak_locs], manage_ticks=False, widths=width,
                           positions=pos, boxprops=dict(facecolor=colors[i],
                                                        fill=True, alpha=0.5),
                           vert=False, showfliers=False, whis=[15, 85],
                           patch_artist=True)

    # plt.title(title)
    plt.ylabel("Z-score")
    plt.xlabel(title + " Latency Time (s)")
    plt.axhline(linestyle='--', color='k')
    plt.axvline(linestyle='--', color='k')
    plt.savefig(cond+'_peaks.svg', dpi=300)

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

    # ## plot on brain
    # pred = np.argmax(W, axis=1)
    # groups = [[sub.keys['channel'][sub.SM[i]] for i in np.where(pred == j)[0]]
    #           for j in range(W.shape[1])]
    # fig1 = sub.plot_groups_on_average(groups,
    #                                   ['blue', 'orange', 'green', 'red'],
    #                                   hemi='lh',
    #                                   rm_wm=False)

    ### plot the weights
    # fig, ax = plt.subplots(1, 1)
    # ax.set_ylim([np.min(W), np.max(W)])

