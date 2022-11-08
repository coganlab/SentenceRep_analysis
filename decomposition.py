import numpy as np
from sklearn.cluster import AgglomerativeClustering, ward_tree
from tslearn.clustering import TimeSeriesKMeans, KShape, silhouette_score
from tslearn.utils import to_sklearn_dataset
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
from tslearn.metrics import gamma_soft_dtw, gak
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.metrics import fowlkes_mallows_score, calinski_harabasz_score, \
    homogeneity_score, completeness_score, v_measure_score
import sklearn.model_selection as ms
from utils.mat_load import get_sigs, load_all, group_elecs
from sklearn.decomposition import NMF
from plotting import plot_opt_k, plot_clustering, alt_plot, plot_weight_dist
from pandas import DataFrame as df
from utils.calc import ArrayLike, BaseEstimator, stitch_mats
import tensorly.decomposition as td
import tensorly as tl

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
    def cv_scorer(estimator: BaseEstimator, X: df, **kwargs):
        if '.decomposition.' in str(estimator.__class__):
            w = estimator.fit_transform(X)
            cluster_labels = np.argmax(w,1)
        else:
            estimator.fit(X)
            cluster_labels = estimator.labels_
        num_labels = len(set(cluster_labels))
        num_samples = len(X.index)
        if num_labels == 1 or num_labels == num_samples:
            return -1
        else:
            return scorer(X, cluster_labels, **kwargs)
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
    test = np.linspace(0, 1, 3)
    param_grid = {'n_components': [4], 'init': ['nndsvda','random'],
                    'solver': ['mu'], 'beta_loss': [2,1,0.5], 'l1_ratio': test,
                    'alpha_W': test, 'alpha_H': test}
    scoring = {#'sil': create_scorer(silhouette_score),
               'calinski': create_scorer(calinski_harabasz_score),
               #'hom': create_scorer(homogeneity_score), 'comp': create_scorer(completeness_score),
               #'v': create_scorer(v_measure_score),
               #'fowlkes': create_scorer(fowlkes_mallows_score)
               }
    # param_dict_sil = {'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    comp = 'n_components'
    # param_dict_sil = {comp: [2, 3, 4, 5],'kernel':['gak','chi2','additive_chi2','rbf','linear','poly','polynomial','laplacian','sigmoid','cosine']}
    gs = ms.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring,
                         cv=cv_ts, n_jobs=-1, verbose=2, return_train_score=True, refit='calinski')
    gs.fit(df(x))
    keys = list(gs.best_estimator_.__dict__.keys())
    thing = keys[comp == keys]
    winner = gs.best_estimator_
    keys = list(gs.best_estimator_.__dict__.keys())
    thing = keys[comp == keys]

    gs.estimator = winner
    # gs.scoring = {'sil': create_scorer(silhouette_score), 'calinski': create_scorer(calinski_harabasz_score)}
    # gs.refit = 'calinski'
    gs.param_grid = {comp: [2, 3, 4, 5, 6]}
    gs.fit(df(x))
    return gs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata_part.mat')
    SM, AUD, PROD = group_elecs(all_sigA, sig_chans)
    #%%
    cond = 'LSwords'
    resp = all_sigA[cond]["Response"]
    respz = all_sigZ[cond]["Response"]
    aud = all_sigA[cond]["AuditorywDelay"]
    audz = all_sigZ[cond]["AuditorywDelay"]
    part = all_sigA[cond]["StartPart"]
    partz = all_sigZ[cond]["StartPart"]
    go = all_sigA[cond]["DelaywGo"]
    goz = all_sigZ[cond]["DelaywGo"]
    stitched = stitch_mats([aud[SM, 0:175], go[SM, :]], [0], axis=1)
    stitchedz = stitch_mats([audz[SM, 0:175], goz[SM, :]], [0], axis=1)

    #%%
    # plt.matshow(sigSum)
    # sumAvg = np.mean(sigSum,axis=0)
    # plt.plot(sumAvg)
    # plt.show()
    #sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans, cond)

    # x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched))
    # gridsearch = estimate(x, NMF(max_iter=100000, tol=1e-8), 3)
    # estimator = gridsearch.best_estimator_
    # estimator.n_components = 4
    # y = estimator.fit_transform(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)))
    # res = df(gridsearch.cv_results_)
    decomp = td.non_negative_parafac
    tens = td.CP_NN_HALS(3, n_iter_max=10000, init='random', exact=True, tol=1e-7)
    tens.mask = tl.tensor(stitched)
    tens.fit(tl.tensor(stitchedz))
    y = tens.decomposition_.factors[0]
    plot_weight_dist(stitchedz, y)

    #
    # gridsearch.scorer_ = gridsearch.scoring = {}
    # np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
    # plt.plot(decomp_sigs)
    # plt.savefig('data/decomp.png')


