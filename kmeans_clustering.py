import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import matplotlib.pyplot as plt

x = np.load("A_LSwords+LMwords_aud+go.npy")
KMeans = TimeSeriesKMeans

def cluster_variance(n):
    variances = []
    kmeans = []
    outputs = []
    K = [i for i in range(1, n + 1)]
    for i in range(1, n + 1):
        variance = 0
        model = KMeans(n_clusters=i, random_state=82, verbose=2).fit(x)
        kmeans.append(model)
        variances.append(model.inertia_)

    return variances, K, n


def calculate_WSS(kmax, repeat=10, points=x):
    sse = []
    for r in range(repeat):
        sse_t=[]
        for k in range(1, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

            sse_t.append(curr_sse)
        sse.append(sse_t)
    sse = np.mean(sse,0)
    return sse


def calc_sil(kmax):
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    sil = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric='euclidean'))
    return sil


def sk_clustering(k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(x)
    #connectivity = nbrs.kneighbors_graph(x)
    ward = AgglomerativeClustering(
        n_clusters=k, linkage="ward"#, connectivity=connectivity
    ).fit(x)
    return ward, nbrs



def plot_clustering(labels, error=False):
    for i in np.unique(labels):
        w_sigs = x[labels == i]
        mean = np.mean(w_sigs,0)
        std = np.std(w_sigs,0)
        tscale = range(np.shape(w_sigs)[1])
        if error:
            plt.errorbar(tscale,mean,std)
        else:
            plt.plot(tscale,mean)
    plt.show()

if __name__ == "__main__":
    n = 20

    # variances, K, n = cluster_variance(30)
    # plt.plot(K, variances)

    sse = calculate_WSS(n, 50)
    plt.plot(sse)
    plt.ylabel("Within-Cluster Sum of Squared Values (WSS)")

    # sil = calc_sil(n)
    # plt.plot(sil)
    # plt.ylabel("silhouette score")
    plt.xlabel("K Value")
    plt.xticks([i for i in range(1, n + 1)])
    plt.show()

    #ward, nbrs = sk_clustering(4)
    #plot_clustering(ward.labels_)
