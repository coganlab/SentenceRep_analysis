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
    aud_slice = (slice(None), slice(0, 175))
    stitched = np.hstack([sub['aud_ls'].sig[aud_slice],
                          sub['aud_lm'].sig[aud_slice],
                          sub['go_ls'].sig, sub['resp'].sig])
    zscores = np.nanmean(sub['zscore'].combine(('stim', 'trial'))._data, axis=-2)
    powers = np.nanmean(sub['power'].combine(('stim', 'trial'))._data, axis=-2)
    plot_data = np.hstack([zscores['aud_ls', :, 0:175], zscores['go_ls']])

    train = np.hstack([powers['aud_ls', :, 0:175],
                       powers['aud_lm', :, 0:175],
                       powers['go_ls'], powers['resp']])
    combined = train * stitched - np.min(train * stitched)

    ## run the decomposition
    scorer = lambda model: calinski_halbaraz(np.array(model.fit.W).T,
                                             np.array(model.fit.H))
    bmf = nimfa.Bmf(stitched[sub.SM], seed="nndsvd", rank=4, max_iter=10000, callback=scorer, track_factor=True, track_error=True)
    bmf_fit = bmf()
    W = np.array(bmf_fit.fit.W)
    this_plot = np.hstack([sub['aud_ls'].sig[aud_slice], sub['go_ls'].sig])
    plot_weight_dist(this_plot[sub.SM], W)
    ## plot on brain
    pred = np.argmax(W, axis=1)
    groups = [[sub.keys['channel'][sub.SM[i]] for i in np.where(pred == j)[0]]
              for j in range(W.shape[1])]
    fig1 = sub.plot_groups_on_average(groups,
                                      ['blue', 'orange', 'green', 'red'],
                                      hemi='lh')

    ##
    lfnmf = nimfa.Lfnmf(combined[sub.SM], seed="nndsvd", rank=4, max_iter=1000)
    lfnmf_fit = lfnmf()
    W = np.array(lfnmf_fit.fit.W)

    # W = NMF(4, init="nndsvda", tol=1e-10, max_iter=10000,
    #         solver='mu').fit_transform(train * stitched - np.min(train * stitched))
    plot_weight_dist(plot_data[sub.SM], W)


    # n = bmf.estimate_rank([2,3,4,5,6,7,8],n_run=100)
    # from MEPONMF.onmf_DA import DA
    # from MEPONMF.onmf_DA import ONMF_DA
    # # k = 10
    # param = dict(tol=1e-8, alpha=1.002,
    #            purturb=0.5, verbos=1, normalize=False)
    # W, H, model = ONMF_DA.func(stitched, k=k, **param, auto_weighting=False)
    # model.plot_criticals(log=True)
    # plt.show()
    # k = model.return_true_number()
    # W, H, model2 = ONMF_DA.func(stitched, k=k, **param, auto_weighting=True)
    # model2 = DA(**param,K=k, max_iter=1000)
    # model2.fit(stitched,Px='auto')
    # Y,P = model2.cluster()
    # model2.plot_criticals(log=True)
    # plt.show()
    # plot_weight_dist(stitchedz, Y)
    # model2.pie_chart()

    # for k in np.array(range(7))+1:
    #     W, H, model = ONMF_DA.func(stitched, k, alpha=1.05, purturb=0.01, tol=1e-7)
    #     # model = DA(k, tol=1e-4, max_iter=1000, alpha=1.05,
    #     #             purturb=0.01, beta_final=None, verbos=0, normalize=False)
    #     # model.fit(stitched, Px='auto')
    #     # y, P = model.cluster()
    #     cost.append(model.return_cost())
    # x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched))
    # gridsearch = estimate(x, NMF(max_iter=100000, tol=1e-8), 3)
    # estimator = gridsearch.best_estimator_
    # estimator = NMF(max_iter=100000,init='nndsvda',alpha_W=0.01,
    #                              alpha_H=0.5, verbose=2)
    # estimator = FactorAnalysis(max_iter=100000,copy=True)
    # test = np.linspace(0, 1, 5)
    # # param_grid = {'n_components': [3], 'init': ['nndsvda'],
    # #               'solver': ['mu'], 'beta_loss': [2, 1, 0.5], 'l1_ratio': test,
    # #               'alpha_W': [0], 'alpha_H': test}
    # param_grid = {'n_components': [4],'rotation' : ['varimax', 'quartimax']}
    # gridsearch = estimate(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)), estimator,
    #                       param_grid, 5)
    # estimator = gridsearch.best_estimator_
    # estimator.n_components = 4
    # y = estimator.fit_transform(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)))
    # res = df(gridsearch.cv_results_)
    # decomp = td.non_negative_parafac
    # tens = td.CP_NN_HALS(3, n_iter_max=10000, init='random', exact=True, tol=1e-7)
    # tens.mask = tl.tensor(stitched)
    # tens.fit(tl.tensor(stitchedz))
    # y = tens.decomposition_.factors[0]
    #k = 4
    #W, H, model = ONMF_DA.func(stitched, k, alpha=1.05, purturb=0.01, tol=1e-4)
    #plot_weight_dist(stitched, W)

    #
    # gridsearch.scorer_ = gridsearch.scoring = {}
    # np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
    # plt.plot(decomp_sigs)
    # plt.savefig('data/decomp.png')
