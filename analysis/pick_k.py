##
import os
import numpy as np
from collections.abc import Sequence
import matplotlib

matplotlib.use('Qt5Agg')

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_dist
import sklearn.decomposition as skd
import sys
from scipy.sparse import csr_matrix, issparse, linalg as splinalg
from numba import njit
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from joblib import Parallel, delayed
import nimfa as nf

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


def calinski_harabasz(X: np.ndarray, W: np.ndarray):
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
    if W.shape[1] == 1:
        return 0
    labels = np.argmax(W, axis=1)
    legal_arr = _check_array(X)
    return calinski_harabasz_score(legal_arr, labels)


def davies_bouldin(X: np.ndarray, W: np.ndarray):
    labels = np.argmax(W, axis=1)
    legal_arr = _check_array(X)
    return davies_bouldin_score(legal_arr, labels)


def silhouette(X: np.ndarray, W: np.ndarray):
    if W.shape[1] == 1:
        return 0
    labels = np.argmax(W, axis=1)
    legal_arr = _check_array(X)
    return silhouette_score(legal_arr, labels)


def explained_variance(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    if not issparse(orig):
        return evar(orig, W, H)
    else:
        diff = orig - csr_matrix(W @ H)
        # sparse_diff = csr_matrix((diff[orig.A.nonzero()], orig.A.nonzero()))
        return 1 - splinalg.norm(diff) ** 2 / splinalg.norm(orig) ** 2


@njit("f8(f8[:,::1], f8[:,::1], f8[::1,:])", nogil=True)
def evar(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return 1 - np.linalg.norm(orig - W @ H) ** 2 / np.linalg.norm(orig) ** 2


def get_k(X: np.ndarray, estimator, k_test: Sequence[int] = range(1, 10),
          metric: callable = explained_variance, n_jobs: int = -3,
          measure: np.ndarray = None, reps: int = 10) -> tuple[matplotlib.pyplot.Axes, np.ndarray]:
    """Estimate the number of components to use for NMF

    Parameters
    ----------
    X : np.ndarray
        The data to fit
    k_test : tuple[int], optional
        The range of k values to test, by default tuple(range(1, 10))
    metric : callable, optional
        The metric to use for estimation, by default explained_variance
    **options
        Additional options to pass to the estimator model

    Returns
    -------
    matplotlib.pyplot.Figure
        The figure containing the plot
    """

    def _repeated_estim(k: int, estimator=estimator
                        ) -> np.ndarray[float]:
        est = []
        for i in range(reps):
            if type(estimator) == skd.NMF:
                estimator.set_params(n_components=k)
                estimator.fit(X)
                Y = estimator.transform(X)
                H = estimator.components_
            elif estimator == 'Psmf':
                estim = nf.Psmf(X, rank=k, n_runs=6, max_iter=1000)
                estim.rank = k
                e = estim()
                Y = e.fit.W.toarray()
                H = e.fit.H.toarray()
            elif measure is not None:
                estimator.fit(X)
                Y = estimator.predict(measure)
                H = estimator.components_
            else:
                Y = estimator.fit_predict(X)
                H = estimator.components_
            est.append(metric(X, Y, H) + (estimator.reconstruction_err_,))
        return np.array(est)

    par_gen = Parallel(n_jobs=n_jobs, verbose=10, return_as='generator')(
        delayed(_repeated_estim)(k) for k in k_test)
    est = np.zeros((reps, len(k_test), 5))
    for i, o in enumerate(par_gen):
        est[:, i, :] = o[...]

    return est


def scale(X, xmax: float = 1, xmin: float = 0):
    return (X - np.min(X)) / (np.max(X) - np.min(X)) * (xmax - xmin) + xmin


if __name__ == "__main__":
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
                          # sub.signif['resp', :]])

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

    ## try clustering
    #
    options = dict(init="random", max_iter=100000, solver='cd', shuffle=False,
                   # beta_loss='kullback-leibler',
                   tol=1e-14, l1_ratio=0.5)
    # model = skd.FastICA(max_iter=1000000, whiten='unit-variance', tol=1e-9)
    # model = skd.FactorAnalysis(max_iter=1000000, tol=1e-9, copy=True,
    #                            svd_method='lapack', rotation='varimax')
    model = skd.NMF(**options)
    # model = tsc.KShape(max_iter=1000000, tol=1e-9, n_clusters=4)
    # model = tsc.TimeSeriesKMeans(n_clusters=4,
    #                              n_jobs=4, metric="dtw", verbose=True)
    idxs = [list(idx & sub.grey_matter) for idx in [sub.AUD, sub.SM, sub.PROD]]
    met_func = lambda X, W, H: (calinski_harabasz(X, W),
                                explained_variance(X, W, H),
                                silhouette(X, W),
                                davies_bouldin(X, W))
    axs = []
    titles = ["Auditory", "Sensorimotor", "Production"]
    fig, axs = plt.subplots(1, 3)
    for idx, ax2 in zip(idxs, axs):
        data = get_k(pval[idx],
                     model,
                     range(2, 10),
                     met_func,
                     n_jobs=6,
                     reps=10)

        minvar = np.min(data[..., 1])

        plot_dist(scale(data[..., 4]-data[..., 0], xmin=minvar), mode='std', times=(2, 9), ax=ax2, label='Calinski')
        plot_dist(data[..., 1], mode='std', times=(2, 9), ax=ax2, label='Explained Variance')
        plot_dist(data[..., 2] + minvar, mode='std', times=(2, 9), ax=ax2, label='Silhouette')
        plot_dist(scale(-data[..., 3], xmin=minvar), mode='std', times=(2, 9), ax=ax2, label='Davies-Bouldin')
        plot_dist(scale(data[..., 4], xmin=minvar), mode='std', times=(2, 9), ax=ax2,
                  label='Reconstruction Error')
        ax2.legend()
        plt.xlabel("K")
        plt.ylabel("Score (A.U. except Explained Variance)")
        ax2.set_title(titles.pop(0))
    fig.suptitle("NMF Clustering Metrics (pvals)")

