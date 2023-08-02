import os
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

from analysis import GroupData
from plotting import plot_weight_dist
import nimfa
from sklearn.decomposition import NMF
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


def calinski_halbaraz(X, W, n_clusters=None):
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

    if issparse(X):
        X = X.toarray()

    if issparse(W):
        W = W.toarray()

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

    ## Load the data
    fpath = os.path.expanduser("~/Box/CoganLab")
    sub = GroupData.from_intermediates("SentenceRep", fpath,
                                       folder='stats')
    ## setup training data
    aud_slice = slice(0, 175)
    stitched = np.hstack([sub['aud_ls'].sig[:, aud_slice],
                          sub['aud_lm'].sig[:, aud_slice],
                          sub['go_ls'].sig, sub['resp'].sig])
    zscores = np.nanmean(sub['zscore'].combine(('stim', 'trial'))._data, axis=-2)
    powers = np.nanmean(sub['power'].combine(('stim', 'trial'))._data, axis=-2)
    plot_data = np.hstack([zscores['aud_ls', :, aud_slice], zscores['go_ls']])

    train = np.hstack([zscores['aud_ls', :, aud_slice],
                       zscores['aud_lm', :, aud_slice],
                       zscores['go_ls'], zscores['resp']])
    combined = train * stitched - np.min(train * stitched)
    raw = train - np.min(train)
    sparse_matrix = csr_matrix((combined[stitched == 1], stitched.nonzero()))


    ## run the decomposition
    ch = [[] for _ in range(9)]
    scorer = lambda model: ch[model.fit.rank-1].append(calinski_halbaraz(
        np.array(model.fit.W).T, np.array(model.fit.H)))
    options = dict(seed="random", rank=4, max_iter=10000,
                   callback=scorer,
                   update='divergence',
                   objective='div',
                   options=dict(flag=0))
    nmf = nimfa.Nmf(sparse_matrix[sub.SM], **options)
    est = nmf.estimate_rank(list(range(1, 10)))
    matplotlib.pyplot.plot([e['evar'] / e['rss'] for e in est.values()])
    ##
    # bmf_fit = bmf()
    # W = np.array(bmf_fit.fit.W)
    # this_plot = np.hstack([sub['aud_ls'].sig[aud_slice], sub['go_ls'].sig])
    # plot_weight_dist(this_plot[sub.SM], W)
    # ## plot on brain
    # pred = np.argmax(W, axis=1)
    # groups = [[sub.keys['channel'][sub.SM[i]] for i in np.where(pred == j)[0]]
    #           for j in range(W.shape[1])]
    # fig1 = sub.plot_groups_on_average(groups,
    #                                   ['blue', 'orange', 'green', 'red'],
    #                                   hemi='lh')

