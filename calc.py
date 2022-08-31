from tslearn.clustering import silhouette_score
from numpy import matlib, array, mean, sqrt, reshape, \
    vstack, sum, outer, argmax, std, shape


def calculate_WSS(centroids, label, points):
    sse = 0

    # calculate square of Euclidean distance of each point from its
    # cluster center and add to current WSS
    for i in range(len(points)):
        curr_center = centroids[label[i]]
        sse += (points[i, 0] - curr_center[0]) ** 2 + \
               (points[i, 1] - curr_center[1]) ** 2

    return sse


def calc_score(X, kmax, model, metric='euclidean'):
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


def dist(mat: array):
    avg = mean(mat, 0)
    avg = reshape(avg, [len(avg)])
    stdev = std(mat, 0) / sqrt(shape(mat)[1])
    stdev = reshape(stdev, [len(stdev)])
    return avg, stdev


def get_elbow(data: array):
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


if __name__ == "__main__":
    from mat_load import load_all, get_sigs
    from kmeans_clustering import plot_opt_k, KMeans
    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans)
    score = plot_opt_k(sigZ['SM'], 10, 20, KMeans(verbose=1, n_init=3), ['euclidean'])
    avg, stdev = dist(score['euclidean']['sil'])
    elb = get_elbow(avg)+1