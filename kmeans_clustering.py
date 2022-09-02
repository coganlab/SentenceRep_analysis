import numpy as np
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape, silhouette_score
import sklearn.decomposition
from sklearn.decomposition import NMF, LatentDirichletAllocation
from tslearn.utils import to_sklearn_dataset
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
from tslearn.metrics import gamma_soft_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from sklearn.metrics import make_scorer, calinski_harabasz_score
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mat_load import get_sigs, load_all
from calc import calc_score, get_elbow, dist, mat_err
from pandas import DataFrame as df


class ts_spectral_clustering(AgglomerativeClustering):
    def __init__(self, **kwargs):
        if 'n_neighbors' in kwargs.keys():
            self.n_neighbors = kwargs.pop('n_neighbors')
        else:
            self.n_neighbors = 5
        if 'metric' in kwargs.keys():
            self.metric = kwargs.pop('metric')
        else:
            self.metric = 'euclidean'
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        knn.fit(X)
        self.connectivity = knn.kneighbors_graph
        params = self.get_params()
        return AgglomerativeClustering(**params).fit(self, X)


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


def plot_clustering(data: np.ndarray, label: np.ndarray,
                    error: bool = False, title: str = None, weighted=False):
    fig, ax = plt.subplots()
    if weighted:
        group = range(min(np.shape(label)))
    else:
        group = np.unique(label)
    for i in group:
        if not weighted:
            w_sigs = data[label == i]
        else:
            try:
                w_sigs = np.array([label[i][j]*dat for j, dat in enumerate(data.T)])
            except (ValueError, IndexError) as e:
                w_sigs = np.array([label.T[i][j]*dat for j, dat in enumerate(data)])
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


def do_decomp(data, clusters=8, repetitions=10, mod=NMF(init='random', max_iter=10000, verbose=2)):
    data = data - np.min(data)
    errs = np.ndarray([repetitions, clusters])
    for k in np.array(range(clusters)):
        mod.set_params(n_components=k+1)
        results = Parallel(-1)(delayed(mat_err)(data, mod) for i in range(repetitions))
        errs[:, k] = results
    mean, std = dist(errs)
    tscale = list(range(len(mean)))
    plt.errorbar(tscale, mean, yerr=std)
    plt.xlabel("K Value")
    plt.xticks(np.array(range(clusters)), np.array(range(clusters)) + 1)
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


def create_scorer(scorer):
    def cv_scorer(estimator, X):
        if '.decomposition.' in str(estimator.__class__):
            w = estimator.fit_transform(X)
            cluster_labels = np.where(w.T == np.max(w.T, 0))[0]
        else:
            estimator.fit(X)
            cluster_labels = estimator.labels_
        num_labels = len(set(cluster_labels))
        num_samples = len(X.index)
        if num_labels == 1 or num_labels == num_samples:
            return -1
        else:
            return scorer(X, cluster_labels)
    return cv_scorer


def main2(sig, metric='euclidean'):
    scores = {}
    models = {}
    for group, x in sig.items():
        scores[group] = plot_opt_k(x, 8, 15, TimeSeriesKMeans(verbose=1, n_init=3), [metric])
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
    cv = [(slice(None), slice(None))]
    cv_ts = ms.TimeSeriesSplit(n_splits=35)
    estimator = NMF(max_iter=10000)
    # estimator = LatentDirichletAllocation(max_iter=10000, learning_method="batch", evaluate_every=2)
    # estimator = AgglomerativeClustering()
    # estimator = KernelKMeans(n_init=10, verbose=2, max_iter=100)
    param_dict_sil = {'n_components': [2, 3, 4, 5, 6], 'init': ['random', 'nndsvd', 'nndsvda', 'nndsvdar'],
                     'solver': ['cd', 'mu'], 'beta_loss': ['frobenius', 'kullback-leibler', 'itakura-saito']}
    # param_dict_sil = {'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    comp = 'n_components'
    # param_dict_sil = {comp: [2, 3, 4, 5],'kernel':['gak','chi2','additive_chi2','rbf','linear','poly','polynomial','laplacian','sigmoid','cosine']}
    gs = ms.GridSearchCV(estimator=estimator, param_grid=param_dict_sil, scoring=create_scorer(silhouette_score),
                         cv=cv_ts, n_jobs=-1, verbose=2, error_score=0, return_train_score=True)
    winners = {}
    x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(sigA['SM']))
    # x = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(x).squeeze()
    # param_dict_sil = {'n_clusters': [2, 3, 4, 5, 6], 'linkage': ['ward', 'complete', 'average', 'single'],
    #                   'connectivity': [
    #                       NearestNeighbors(n_neighbors=i + 1, metric='euclidean', n_jobs=-1, verbose=2).fit(
    #                           sig[group]).kneighbors_graph for i in range(10)]}
    gs.fit(df(x),None)
    keys = list(gs.best_estimator_.__dict__.keys())
    thing = keys[comp == keys]
    if gs.best_estimator_.__dict__[thing] == 2:
        gs.param_grid[comp] = [1, 2, 3, 4, 5, 6]
        gs.scoring = create_scorer(calinski_harabasz_score)
        gs.fit(df(x),None)
    winner = gs.best_estimator_
    gs.estimator = winner
    gs.param_grid = {comp: [i+1 for i in range(winner.__dict__[thing]+2)]}
    for group, x in sigA.items():
        gs.fit(df(x))
        winners[group] = gs.best_estimator_
        w = winners[group].fit_transform(x)
        plot_clustering(x, w, True, group, True)
