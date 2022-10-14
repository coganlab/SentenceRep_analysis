import numpy as np
from tslearn.clustering import silhouette_score
from numpy import matlib, array, mean, sqrt, reshape, concatenate, \
    vstack, sum, outer, argmax, std, shape, linalg, ndarray, min, multiply, \
    linspace
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from typing import Union
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator


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


def dist(mat: ArrayLike):
    avg = mean(mat, 0)
    avg = reshape(avg, [len(avg)])
    stdev = std(mat, 0) / sqrt(shape(mat)[1])
    stdev = reshape(stdev, [len(stdev)])
    return avg, stdev


def mat_err(data: ArrayLike, mod: BaseEstimator):
    W = mod.fit_transform(data)
    H = mod.components_
    error = linalg.norm(data - W @ H) ** 2 / linalg.norm(data) ** 2
    return error


def get_elbow(data: ArrayLike):
    """ Draws a line between the first and last points in a dataset and finds the point furthest from that line.

    :param data:
    :return:
    """
    nPoints = len(data)
    allCoord = vstack((range(nPoints), data)).T
    array([range(nPoints), data])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / sqrt(sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = sum(vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = sqrt(sum(vecToLine ** 2, axis=1))
    # set distance to points below lineVec to 0
    distToLine[vecToLine[:, 1] < 0] = 0
    idxOfBestPoint = argmax(distToLine)
    return idxOfBestPoint


def stitch_mats(mats: list[array], overlaps: list[int], axis: int = 0) -> array:
    """break up the matrices into their overlapping and non-overlapping parts then stitch them back together
    :param mats: list of matrices to stitch together
    :param overlaps: list of the number of overlapping rows between each matrix
    :param axis: axis to stitch along
    :return: stitched matrix
    """
    stitches = [mats[0]]
    if len(mats) != len(overlaps) + 1:
        raise ValueError("The number of matrices must be one more than the number of overlaps")
    for i, over in enumerate(overlaps):
        stitches = stitches[:-2] + merge(stitches[-1], mats[i+1], over, axis)
    return concatenate(stitches, axis=axis)


def merge(mat1: array, mat2: array, overlap: int, axis: int = 0) -> list[array]:
    """Take two arrays and merge them over the overlap gradually"""
    sl = [slice(None)] * mat1.ndim
    sl[axis] = slice(0, mat1.shape[axis]-overlap)
    start = mat1[sl]
    sl[axis] = slice(mat1.shape[axis]-overlap, mat1.shape[axis])
    middle1 = multiply(linspace(1, 0, mat1.shape[axis]), mat1[sl])
    sl[axis] = slice(0, overlap)
    middle2 = multiply(linspace(0, 1, mat2.shape[axis]), mat2[sl])
    middle = sum(middle1, middle2)
    sl[axis] = slice(overlap, mat2.shape[axis])
    last = mat2[sl]
    return [start, middle, last]


if __name__ == "__main__":
    from mat_load import load_all, get_sigs
    from decomposition import plot_opt_k, KMeans

    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('../data/pydata.mat')
    sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans)
    score = plot_opt_k(sigZ['SM'], 10, 20, KMeans(verbose=1, n_init=3), ['euclidean'])
    avg, stdev = dist(score['euclidean']['sil'])
    elb = get_elbow(avg) + 1
