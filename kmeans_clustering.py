import numpy as np
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape, silhouette_score
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
from tslearn.metrics import gamma_soft_dtw
from sklearn.metrics import make_scorer
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mat_load import get_sigs, load_all
from calc import calc_score, get_elbow, dist


def sk_clustering(x,k,metric='euclidean'):
    kwargs = dict()
    if metric == 'softdtw':
        kwargs['gamma'] = gamma_soft_dtw(x)
    nbrs = NearestNeighbors(n_neighbors=int(np.shape(x)[0]/k/2), n_jobs=-1,
                            metric=metric, metric_params=kwargs)
    nbrs.fit(x)
    connectivity = nbrs.kneighbors_graph(x)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
    ward.fit(x)
    return ward, nbrs


def do_nmf(data):
    X = data - np.min(data)
    errs = []
    for i in range(10):
        err = []
        for k in np.array(range(10)) + 1:
            model2 = NMF(n_components=k, init='random', max_iter=10000)
            W = model2.fit_transform(X)
            H = model2.components_
            err.append(np.linalg.norm(X - W @ H) ** 2 / np.linalg.norm(X) ** 2)
        errs.append(err)
    mean, std, tscale, = dist(errs)
    plt.errorbar(tscale, mean, yerr=std)
    plt.show()


def plot_clustering(data: np.ndarray, label: np.ndarray,
                    error: bool = False, title: str = None, weighted=False):
    fig, ax = plt.subplots()
    if weighted:
        group = range(np.shape(label)[0])
    else:
        group = np.unique(label)
    for i in group:
        if not weighted:
            w_sigs = data[label == i]
        else:
            w_sigs = np.array([label[i][j]*dat for j, dat in enumerate(data.T)])
        mean, std = dist(w_sigs)
        tscale = range(len(mean))
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


def plot_opt_k(data: np.array, n, rep, model, methods=None, title=None):
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
        for key, value in score.items():
            mean, std = dist(value)
            tscale = range(len(mean))
            plt.errorbar(tscale, mean, yerr=std)
            plt.ylabel(method + key)
            plt.xlabel("K Value")
            plt.xticks(np.array(range(n)), np.array(range(n)) + 1)
            plt.title(title)
            plt.show()
        score['k'] = get_elbow(np.mean(sil, 0)) + 1
        results[method] = score
    return results


def alt_plot(X_train, y_pred):
    plt.figure()
    for yi in range(len(np.unique(y_pred))):
        plt.subplot(len(np.unique(y_pred)), 1, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.xlim(0, X_train.shape[1])
        plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    plt.show()


def main(sig, metric='euclidean'):
    scores = {}
    models = {}
    for group, x in sig.items():
        scores[group] = plot_opt_k(x, 8, 15, TimeSeriesKMeans(verbose=1, n_init=3), [metric])
        # models[group] = list(range(3))
        # x = x.T

        kwargs = {}
        kwargs['metric'] = metric
        kwargs['n_jobs'] = -1
        models[group] = TimeSeriesKMeans(n_clusters=scores[group][metric]['k'],
                                         n_init=10, verbose=2, **kwargs)
        models[group].fit(x)
        weights = models[group].cluster_centers_.squeeze()
        labels = models[group].predict(x)
        # labels = np.where((weights == np.max(weights, 0)).T)[1]
        plot_clustering(x, labels, True, group)
        # plot_clustering(x, weights, True, group,True)
        dat = sigZ[group]
        dat[dat>=1] = sigZ[group][dat>=1]
        alt_plot(dat, labels)
    return scores, models


if __name__ == "__main__":
    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans)
    scores, models = main(sigZ,'softdtw')
    scores2, models2 = main(sigA,'softdtw')
    ts_cv = ms.TimeSeriesSplit(n_splits=5)

    knn = NearestNeighbors()
