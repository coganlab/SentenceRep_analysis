import numpy as np
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.decomposition import NMF
from tslearn.clustering import silhouette_score, KernelKMeans, KShape
from tslearn.clustering import TimeSeriesKMeans as KMeans
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
from tslearn.barycenters import softdtw_barycenter
import matplotlib.pyplot as plt

SM = np.load('../Z_LSwords_aud+resp_SM.npy')
AUD = np.load('../Z_LSwords_aud+resp_AUD.npy')
PROD = np.load('../Z_LSwords_aud+resp_PROD.npy')
sigA = {'SM': np.load('../A_LSwords_aud+resp_SM.npy'),
        'AUD': np.load('../A_LSwords_aud+resp_AUD.npy'),
        'PROD': np.load('../A_LSwords_aud+resp_PROD.npy')}


def cluster_variance(n):
    variances = []
    kmeans = []
    outputs = []
    K = [i for i in range(1, n + 1)]
    for i in range(1, n + 1):
        variance = 0
        model = KMeans(n_clusters=i, verbose=2).fit(x)
        kmeans.append(model)
        variances.append(model.inertia_)

    return variances, K


def calculate_WSS(centroids, labels, points):
    sse = 0

    # calculate square of Euclidean distance of each point from its
    # cluster center and add to current WSS
    for i in range(len(points)):
        curr_center = centroids[labels[i]]
        sse += (points[i, 0] - curr_center[0]) ** 2 + \
            (points[i, 1] - curr_center[1]) ** 2

    return sse


def calc_sil(kmax, metric='euclidean'):
    # dissimilarity would not be defined for a single cluster, thus, minimum
    #  number of clusters should be 2
    sil = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(k, metric=metric, n_init=5, n_jobs=-1, verbose=2)
        labels = kmeans.fit_predict(x)
        sil.append(silhouette_score(x, labels, metric=metric))
    sil = [sil[0]] + sil
    return sil


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
            sil.append(silhouette_score(X, labels, metric=metric))
        var.append(model.inertia_)
        wss.append(calculate_WSS(model.cluster_centers_,labels,X))
    sil = [sil[0]] + sil
    return sil, var, wss


def sk_clustering(k):
    nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(x)
    connectivity = nbrs.kneighbors_graph(x)
    ward = AgglomerativeClustering(
        n_clusters=k, linkage="ward", connectivity=connectivity
    ).fit(x)
    return ward, nbrs


def plot_clustering(x, labels, error=False):
    for i in np.unique(labels):
        w_sigs = x[labels == i]
        if error:
            plot_std(w_sigs)
        else:
            mean = np.mean(w_sigs, 0)
            tscale = range(np.shape(w_sigs)[1])
            plt.plot(tscale, mean)
    plt.show()


def plot_std(mat: np.array):
    mean = np.mean(mat, 0)
    mean = np.reshape(mean, [len(mean)])
    std = np.std(mat, 0) / np.sqrt(np.shape(mat)[1])
    std = np.reshape(std, [len(std)])
    tscale = range(np.shape(mat)[1])
    plt.errorbar(tscale, mean, yerr=std)


def plot_opt_k(X, n, rep):
    model = KMeans(n_jobs=-1, verbose=2)
    methods = ['euclidean', 'dtw', 'softdtw']
    scores = {}
    for method in methods:
        model.set_params(**{'metric': method})
        sil = list(range(rep))
        var = list(range(rep))
        wss = list(range(rep))
        for i in range(rep):
            sil[i], var[i], wss[i] = calc_score(X, n, model, method)
        score = {'sil': sil, 'var': var, 'wss': wss}
        for key, value in score.items():
            plot_std(value)
            plt.ylabel("score")
            plt.xlabel("K Value")
            plt.xticks(np.array(range(n)), np.array(range(n))+1)
            plt.title(method+"_"+key+"_"+str(len(X)))
            plt.show()
        scores[method] = score
    return scores


if __name__ == "__main__":

    for x,k in zip([SM, AUD, PROD],[4,4,3]):
        scores = plot_opt_k(x, 10, 10)
        model = KMeans(n_clusters=k, metric='softdtw', n_init=3, n_jobs=-1, verbose=2)
        labels = model.fit_predict(x)
        plot_clustering(x, labels, error=True)

    #model = KMeans(n_clusters=4, metric='softdtw', n_init=4, n_jobs=-1, verbose=2)
    # = KernelKMeans(n_clusters=4, n_jobs=-1, kernel="laplacian", verbose=2)
    #model = KShape(n_clusters=4, verbose=2)
    # model2 = NMF(n_components=4, init='random')
    # W = model2.fit_transform(x)
    # H = model2.components_
    # model, nbrs = sk_clustering(4)
    #model.fit(x)
    #labels = model.fit_predict(x)
    #plot_clustering(labels, error=True)

    #soft_dtw_graph(SM)
