import numpy as np
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from tslearn.clustering import TimeSeriesKMeans as KMeans
from sklearn.decomposition import NMF
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mat_load import Task, sigZ, sigA, sigChans, sigMatChansLoc, sigMatChansName, Subject, SM, AUD, PROD
from calc import calc_score, get_elbow, dist

allsigZ = sigZ
allsigA = sigA
sigZ = {'SM': np.load('data/Z_LSwords_aud+go_SM.npy'),
        'AUD': np.load('data/Z_LSwords_aud+go_AUD.npy'),
        'PROD': np.load('data/Z_LSwords_aud+go_PROD.npy')}
sigA = {'SM': np.load('data/A_LSwords_aud+go_SM.npy'),
        'AUD': np.load('data/A_LSwords_aud+go_AUD.npy'),
        'PROD': np.load('data/A_LSwords_aud+go_PROD.npy')}


def sk_clustering(k):
    nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(x)
    connectivity = nbrs.kneighbors_graph(x)
    ward = AgglomerativeClustering(
        n_clusters=k, linkage="ward", connectivity=connectivity
    ).fit(x)
    return ward, nbrs


def do_nmf(data):
    X = data - np.min(data)
    errs = []
    for i in range(10):
        err = []
        for k in np.array(range(10))+1:
            model2 = NMF(n_components=k, init='random', max_iter=10000)
            W = model2.fit_transform(X)
            H = model2.components_
            err.append(np.linalg.norm(X - W @ H) ** 2 / np.linalg.norm(X) ** 2)
        errs.append(err)
    mean, std, tscale, = dist(errs)
    plt.errorbar(tscale, mean, yerr=std)
    plt.show()


def plot_clustering(data: np.ndarray, label: np.ndarray,
                    error: bool = False, title: str = None):
    fig, ax = plt.subplots()
    for i in np.unique(label):
        w_sigs = data[labels == i]
        mean, std, tscale, = dist(w_sigs)
        if error:
            ax.errorbar(tscale, mean, yerr=std)
        else:
            ax.plot(tscale, mean)
        # the x coords of this transformation are data, and the
        # y coord are axes
    trans = ax.get_xaxis_transform()
    ax.text(50, 0, 'aud onset', rotation=90, transform=trans)
    ax.axvline(175)
    ax.text(225, 0, 'go cue', rotation=90, transform=trans)
    if title is not None:
        plt.title(title)
    plt.show()


def par_calc(data, n, rep, model, method):
    sil = np.ndarray([rep, n])
    var = np.ndarray([rep, n])
    wss = np.ndarray([rep, n])
    results = Parallel(-1)(delayed(calc_score)(data, n, model, method) for i in range(rep))
    for i, result in enumerate(results):
        sil[i, :], var[i, :], wss[i, :] = result
    return sil, var, wss


def plot_opt_k(data, n, rep, model, methods=None, title=None):
    if methods is None:
        methods = ['euclidean', 'dtw', 'softdtw']
    if title is None:
        title = str(len(data))
    title = "Optimal k for " + title
    results = {}
    for method in methods:
        model.metric = method
        sil, var, wss = par_calc(data, n, rep, model, method)
        score = {'sil': sil, 'var': var, 'wss': wss}
        for key, value in {'sil': sil}.items():
            mean, std, tscale, = dist(value)
            plt.errorbar(tscale, mean, yerr=std)
            plt.ylabel(method + " Silhouette Score")
            plt.xlabel("K Value")
            plt.xticks(np.array(range(n)), np.array(range(n)) + 1)
            plt.title(title)
            plt.show()
        score['k'] = get_elbow(np.mean(sil, 0)) + 1
        results[method] = score
    return results


if __name__ == "__main__":
    scores = {}
    models = {}
    for group, x in sigZ.items():
        scores[group] = plot_opt_k(x, 10, 20, KMeans(verbose=1, n_init=3), ['euclidean'])
        models[group] = KMeans(n_clusters=scores[group]['euclidean']['k'],
                               metric='euclidean', n_init=10, n_jobs=-1, verbose=2)
        labels = models[group].fit_predict(x)
        plot_clustering(sigZ[group], labels, True, group)

    # model = KMeans(n_clusters=4, metric='softdtw', n_init=4, n_jobs=-1, verbose=2)
    # = KernelKMeans(n_clusters=4, n_jobs=-1, kernel="laplacian", verbose=2)
    # model = KShape(n_clusters=4, verbose=2)
    # X = SM - np.min(SM)
    # errs = []
    # for i in range(10):
    #     err = []
    #     for k in np.array(range(10))+1:
    #         model2 = NMF(n_components=k, init='random', max_iter=10000)
    #         W = model2.fit_transform(X)
    #         H = model2.components_
    #         err.append(np.linalg.norm(X - W @ H) ** 2 / np.linalg.norm(X) ** 2)
    #     errs.append(err)
    # plot_std(errs)
    # plt.show()

    # model, nbrs = sk_clustering(4)
    # model.fit(x)
    # labels = model.fit_predict(x)
    # plot_clustering(labels, error=True)

    # soft_dtw_graph(SM)
