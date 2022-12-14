import numpy as np
from sklearn.cluster import AgglomerativeClustering, ward_tree
from tslearn.clustering import TimeSeriesKMeans, KShape, silhouette_score
from tslearn.utils import to_sklearn_dataset
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
from tslearn.metrics import gamma_soft_dtw, gak
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.metrics import make_scorer, calinski_harabasz_score
import sklearn.model_selection as ms
from utils.mat_load import get_sigs, load_all, group_elecs
from sklearn.decomposition import NMF
from plotting import plot_opt_k, plot_clustering, alt_plot
from pandas import DataFrame as df
from utils.calc import ArrayLike, BaseEstimator


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


def sk_clustering(x: ArrayLike, k: int, metric: str = 'euclidean'):
    kwargs = dict()
    if metric == 'softdtw':
        kwargs['gamma'] = gamma_soft_dtw(x)
    nbrs = NearestNeighbors(n_neighbors=int(np.shape(x)[0] / k / 2), n_jobs=-1,
                            metric=metric, metric_params=kwargs)
    nbrs.fit(x)
    connectivity = nbrs.kneighbors_graph(x)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
    ward.fit(x)
    return ward, nbrs


def create_scorer(scorer):
    def cv_scorer(estimator: BaseEstimator, X: ArrayLike):
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


def main2(sig: dict[str, ArrayLike], metric: str = 'euclidean'):
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
        dat[dat >= 1] = sigZ[group][dat >= 1]
        alt_plot(dat, labels)
    return scores, models


def estimate(x: ArrayLike, estimator: BaseEstimator, splits: int = 5):
    cv = [(slice(None), slice(None))]
    cv_ts = ms.TimeSeriesSplit(n_splits=splits)
    # estimator = LatentDirichletAllocation(max_iter=10000, learning_method="batch", evaluate_every=2)
    # estimator = AgglomerativeClustering()
    # estimator = KernelKMeans(n_init=10, verbose=2, max_iter=100)
    test = np.linspace(0, 1, 5)
    param_grid = {'n_components': [2, 3, 4], 'init': ['random', 'nnsvd', 'nndsvda'],
                    'solver': ['cd'], 'beta_loss': [2], 'l1_ratio': [0]}
    # scoring = {'sil': create_scorer(silhouette_score), 'calinski': create_scorer(calinski_harabasz_score)}
    # param_dict_sil = {'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    comp = 'n_components'
    # param_dict_sil = {comp: [2, 3, 4, 5],'kernel':['gak','chi2','additive_chi2','rbf','linear','poly','polynomial','laplacian','sigmoid','cosine']}
    gs = ms.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=create_scorer(calinski_harabasz_score),
                         cv=cv_ts, n_jobs=-1, verbose=2, error_score=0, return_train_score=True)
    gs.fit(df(x))
    keys = list(gs.best_estimator_.__dict__.keys())
    thing = keys[comp == keys]
    winner = gs.best_estimator_
    # keys = list(gs.best_estimator_.__dict__.keys())
    # thing = keys[comp == keys]

    # gs.estimator = winner
    # gs.scoring = {'sil': create_scorer(silhouette_score), 'calinski': create_scorer(calinski_harabasz_score)}
    # gs.refit = 'calinski'
    # gs.param_grid = {comp: [i + 2 for i in range(winner.__dict__[thing] + 2)]}
    # gs.fit(df(x))
    return gs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/whole.mat')
    #SM, AUD, PROD = group_elecs(all_sigA, sig_chans)
    cond = 'LSwords'
    aud = all_sigA[cond]["AuditoryWhole"]
    go = all_sigA[cond]["GoWhole"]
    resp = all_sigA[cond]["ResponseWhole"]
    newSet = [aud, go, resp]
    sigConcat = np.concatenate(newSet,axis=1)
    nonActive = np.where(np.all(np.isclose(sigConcat, 0), axis=1))
    newSet.append(sigConcat)
    for i, allign in enumerate(newSet):
        newSet[i] = np.delete(allign, nonActive, axis=0)
    [aud, go, resp, sigConcat] = newSet[:]
    sigSum = np.sum(np.array(newSet[0:3]),axis=0)
    plt.imshow(sigSum)
    sumAvg = np.mean(sigSum,axis=0)
    plt.plot(sumAvg)
    # plt.show()
    #sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans, cond)

    # x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 3)).fit_transform(sigSum))
    # gridsearch = estimate(x, NMF(max_iter=100000), 2)
    # estimator = gridsearch.best_estimator_
    # y = estimator.fit_transform(x)
    # decomp_sigs = np.dot(x.T,y)
    #
    # gridsearch.scorer_ = gridsearch.scoring = {}
    # np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
    # plt.plot(decomp_sigs)
    # plt.savefig('data/decomp.png')


