# from tslearn.clustering import silhouette_score
from numpy import matlib, array, sqrt, reshape, concatenate, vstack, add,\
    outer, argmax, std, shape, linalg, ndarray, min, multiply, linspace, ones,\
    sum, divide, log, pi, zeros, mean, newaxis
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from typing import Union
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances


def calculate_WSS(centroids: ArrayLike, label: list, points: ArrayLike) -> ArrayLike:
    sse = 0
    # calculate square of Euclidean distance of each point from its
    # cluster center and add to current WSS
    for i in range(len(points)):
        curr_center = centroids[label[i]]
        sse += (points[i, 0] - curr_center[0]) ** 2 + \
               (points[i, 1] - curr_center[1]) ** 2
    return sse


def do_decomp(data: ArrayLike, clusters: int = 8, repetitions: int = 10,
              mod: BaseEstimator = NMF(init='random', max_iter=10000, verbose=2)) -> ArrayLike:
    data = data - min(data)
    errs = ndarray([repetitions, clusters])
    for k in array(range(clusters)):
        mod.set_params(n_components=k + 1)
        results = Parallel(-1)(delayed(mat_err)(data, mod) for i in range(repetitions))
        errs[:, k] = results
    return errs


def weight2label(w: ArrayLike) -> Union[list, ndarray]:
    myshape = shape(w)
    if myshape[0] > myshape[1]:
        w = w.T
    return argmax(w, axis=0)


def par_calc(data: ArrayLike, n: int, rep: int, model: BaseEstimator, method: str
             ) -> tuple[ndarray, ndarray, ndarray]:
    sil = ndarray([rep, n])
    var = ndarray([rep, n])
    wss = ndarray([rep, n])
    results = Parallel(-1)(delayed(calc_score)(data, n, model, method) for i in range(rep))
    for i, result in enumerate(results):
        sil[i, :], var[i, :], wss[i, :] = result
    return sil, var, wss


def calc_score(X: ArrayLike, kmax: int, model: BaseEstimator, metric='euclidean'):
    sil = []
    var = []
    wss = []
    for k in range(1, kmax + 1):
        model.set_params(n_clusters=k)
        labels = model.fit_predict(X)
        if k == 1:
            pass
        else:
            sil.append(float(silhouette_score(X, labels, metric=metric, n_jobs=-1)))
        var.append(float(model.inertia_))
        wss.append(float(calculate_WSS(model.cluster_centers_, labels, X)))
    sil = [sil[0]] + sil
    return sil, var, wss


def bic_onmf(X, W, H, k):
    """
    Here, X is the original data matrix, W and H are the non-negative factor
    matrices from the ONMF decomposition, and k is the number of factors
    used in the decomposition. The function first computes the residual sum
    of squares (RSS) as the sum of squared errors between the original data
    and the reconstructed data from the ONMF factorization. It then
    estimates the error variance sigma2 as the ratio of the RSS to the
    number of elements in the data matrix, and computes the log-likelihood
    of the data given the factorization using the formula described in the
    previous answer. Finally, it calculates the BIC using the formula -2 *
    log(L) + k * log(n * p) and returns the result.
    """
    n, p = X.shape
    RSS = sum((X - W @ H.T)**2)
    sigma2 = RSS / (n * p)
    L = -0.5 * (n * p * log(2 * pi * sigma2) + n * p * log(RSS / n))
    bic = -2 * L + k * log(n * p)
    return bic


def ch_onmf(X, W, n_clusters=None):
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
    centroids = zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        centroids[i] = mean(X[W[:, i] > 0], axis=0)

    # Compute the total sum of squares
    # TSS = sum(pairwise_distances(X, mean(X, axis=0, keepdims=True))**2)

    # Compute the between-cluster sum of squares
    BSS = sum(sum(W[:, i, newaxis] * pairwise_distances(centroids[i, newaxis, :], mean(X, axis=0, keepdims=True))**2) for i in range(n_clusters))

    # Compute the within-cluster sum of squares
    WSS = sum(sum(W[:, i, newaxis] * pairwise_distances(X, centroids[i, newaxis, :])**2) for i in range(n_clusters))

    # Compute the CH index
    ch = (BSS / (n_clusters - 1)) / (WSS / (n_samples - n_clusters))

    return ch


def mat_err(data: ArrayLike, mod: BaseEstimator) -> float:
    W = mod.fit_transform(data)
    H = mod.components_
    error = linalg.norm(data - W @ H) ** 2 / linalg.norm(data) ** 2
    return error


if __name__ == "__main__":
    from mat_load import load_all, get_sigs
    from decomposition import plot_opt_k, KMeans

    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('../data/pydata.mat')
    sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans)
    score = plot_opt_k(sigZ['SM'], 10, 20, KMeans(verbose=1, n_init=3), ['euclidean'])
    avg, stdev = dist(score['euclidean']['sil'])
    elb = get_elbow(avg) + 1
