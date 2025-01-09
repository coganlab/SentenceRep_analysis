##
import os
import numpy as np
from collections.abc import Sequence
import matplotlib

matplotlib.use('Qt5Agg')

from analysis.grouping import GroupData
from ieeg.viz.ensemble import plot_dist, plot_weight_dist, subgrids
from ieeg.arrays.label import LabeledArray
import sklearn.decomposition as skd
import sys
from scipy.sparse import csr_matrix, issparse, linalg as splinalg
import scipy.stats as st
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


# @njit("f8(f8[:,::1], f8[:,::1], f8[:,:])", nogil=True)
def evar(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return 1 - np.linalg.norm(orig - W @ H) ** 2 / np.linalg.norm(orig) ** 2


def reconstruction_error(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(orig - W @ H)


def orthogonality(W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(W.T @ W - np.eye(W.shape[1]))


def sparsity(W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(W, ord=1) + np.linalg.norm(H, ord=1)


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
        W = []
        for i in range(reps):
            if type(estimator) == skd.NMF:
                estimator.set_params(n_components=k)
                estimator.fit(X)
                Y = estimator.transform(X)
                H = estimator.components_
            elif type(estimator) == skd.DictionaryLearning:
                estimator.set_params(n_components=k)
                estimator.fit(X)
                Y = estimator.transform(X)
                H = estimator.components_
                estimator.reconstruction_err_ = np.linalg.norm(X - Y @ H)
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
            try:
                est.append(metric(X, Y, H) + (estimator.reconstruction_err_,))
                W.append(Y)
            except ValueError:
                est.append(np.array([np.nan] * 5))
        return np.array(est), W

    par_gen = Parallel(n_jobs=n_jobs, verbose=10, return_as='generator')(
        delayed(_repeated_estim)(k) for k in k_test)
    est = np.zeros((reps, len(k_test), 5))

    Ws = []
    for i, (o, W) in enumerate(par_gen):
        est[:, i, :] = o[...]
        Ws.append(W)
    return est, Ws


def scale(X, xmax: float = 1, xmin: float = 0):
    return (X - np.min(X)) / (np.max(X) - np.min(X)) * (xmax - xmin) + xmin


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig1, axs1 = subgrids(5, 2, 3, (0,))
    fig2, axs2 = subgrids(5, 2, 3, (0,))
    fig3, axs3 = subgrids(5, 2, 3, (0,))
    fig4, axs4 = subgrids(5, 2, 3, (0,))
    fig5, axs5 = subgrids(5, 2, 3, (0,))
    axss = [axs1, axs2, axs3, axs4, axs5]
    suptitles = ["calinski_harabasz", "explained_variance",
                 "silhouette", "davies_bouldin", "err"]
    for fig, title in zip([fig1, fig2, fig3, fig4, fig5], suptitles):
        fig.suptitle(title)

    kwarg_sets = [dict(folder='stats'),
                  dict(folder='stats')]
    fnames = ["short", "wide"]
    titles = ["AUD", "SM", "PROD", "ALL"]
    colors = ['green', 'red', 'blue', 'grey']
    conds = {'aud_ls': (-0.5, 1.5), 'go_ls': (-0.5, 1.5), 'resp': (-1, 1)}
    met_func = lambda X, W, H: (calinski_harabasz(X, W),
                                explained_variance(X, W, H),
                                silhouette(X, W),
                                davies_bouldin(X, W)*-1)
    for i, kwargs in enumerate(kwarg_sets):
        ## Load the data
        fpath = os.path.expanduser("~/Box/CoganLab")
        sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs)
        ## setup training data
        aud_slice = slice(0, 175)
        pval = np.where(sub.p_vals > 0.9999, 0.9999, sub.p_vals)

        # pval[pval<0.0001] = 0.0001
        zscores = LabeledArray(st.norm.ppf(1 - pval), sub.p_vals.labels)
        powers = np.nanmean(sub['zscore'].array, axis=(-4, -2))
        met = powers
        trainp = np.hstack([met['aud_ls', :, aud_slice],
                            met['aud_lm', :, aud_slice],
                            # met['aud_jl', :, aud_slice],
                            met['go_ls'],
                            # met['go_lm'],
                            # met['go_jl'],
                            met['resp']])
                            # powers['resp']])
        trainp -= np.min(trainp)
        # raw = train - np.min(train)
        stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                              # sub.signif['aud_lm', :, aud_slice],
                              sub.signif['resp', :]])
        # sparse_matrix = csr_matrix((trainp[stitched == 1], stitched.nonzero()))
        # sparse_matrix.data -= np.min(sparse_matrix.data)

        ## try clustering
        #
        # model = skd.FastICA(max_iter=1000000, whiten='unit-variance', tol=1e-9)
        # model = skd.FactorAnalysis(max_iter=1000000, tol=1e-9, copy=True,
        #                            svd_method='lapack', rotation='varimax')
        model = skd.NMF(init="random", max_iter=100000, solver='mu', shuffle=False,
                       l1_ratio=1,
                       # beta_loss='kullback-leibler',
                       )
        # model = skd.DictionaryLearning(positive_dict=True, n_jobs=7, verbose=10)
        # model = tsc.KShape(max_iter=1000000, tol=1e-9, n_clusters=4)
        # model = tsc.TimeSeriesKMeans(n_clusters=4,
        #                              n_jobs=4, metric="dtw", verbose=True)
        idxs = [list(idx & sub.grey_matter) for idx in [sub.AUD, sub.SM, sub.PROD]]
        if i == 0:
            idxs.append(list(sub.sig_chans))
        ranks = list(range(len(idxs)))
        for j, idx in enumerate(idxs):

            data, W = get_k(trainp[idx],
                         model,
                         range(2, 9),
                         met_func,
                         n_jobs=7,
                         reps=30)

            minvar = np.min(data[..., 1])
            for k, axs in enumerate(axss):
                ax = axs[0][i]
                if i + k == 0:
                    ax.get_figure().suptitle(titles[j])
                plot_dist(np.max(data[..., k], axis=0)[None], mode='std', times=(2,8), ax=ax,
                          label=titles[j], color=colors[j])
                loc = np.unravel_index(np.argmax(data[..., k]), data.shape[:-1])
                ranks[j] = loc[1] + 2
                ax.legend()
                ax.set_ylabel("Score")
                ax.set_xlabel("K")
                # %% plot the best model
                for x, cond in enumerate(conds):

                    ax = axs[j + 1][i][x]
                    plot_weight_dist(zscores[cond, idx].__array__(),
                                     W[loc[1]][loc[0]], times=conds[cond],
                                     sig_titles=titles, ax=ax)
                    if i + x == 0:
                        ax.set_ylabel(titles[j])

