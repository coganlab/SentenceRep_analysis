import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tslearn.clustering import silhouette_score
from tslearn.clustering import TimeSeriesKMeans as KMeans
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
import matplotlib.pyplot as plt

x = np.load("../A_LSwords_aud+go_SM.npy")


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


def calculate_WSS(kmax, points=x):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


def calc_sil(kmax):
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    sil = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric='euclidean'))
    return sil


def calc_score(kmax, method="sil", repeat=10):
    k_scores = []
    for r in range(repeat):
        if method.lower() in ["sil", "silhouette"]:
            score = calc_sil(kmax)
        elif method.lower() == "wss":
            score = calculate_WSS(kmax)
        elif method.lower() in ["var", "variance"]:
            [score, _] = cluster_variance(kmax)
        else:
            raise SyntaxError("Scoring options are currently 'sil', 'var', and 'wss'")
        k_scores.append(score)
    scores = np.mean(k_scores, 0)
    return scores, k_scores


def sk_clustering(k):
    nbrs = NearestNeighbors(n_neighbors=2, n_jobs=8).fit(x)
    connectivity = nbrs.kneighbors_graph(x)
    ward = AgglomerativeClustering(
        n_clusters=k, linkage="ward", connectivity=connectivity
    ).fit(x)
    return ward, nbrs


def plot_clustering(labels, error=False):
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


if __name__ == "__main__":

    n = 20
    rep = 20
    for s_type in ['wss', 'sil', 'var']:
        _, scores = calc_score(n, s_type, rep)
        plot_std(scores)
        # plt.scatter(range(n), scores)
        plt.ylabel("score")
        plt.xlabel("K Value")
        plt.xticks(np.array(range(n))+1)
        plt.title(s_type)
        plt.show()

    #model = KMeans(n_clusters=4, verbose=2).fit(x)
    #model, nbrs = sk_clustering(4)
    #plot_clustering(model.labels_, error=True)
