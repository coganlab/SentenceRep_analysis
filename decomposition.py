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
from sklearn.decomposition import NMF, FactorAnalysis
from plotting import plot_opt_k, plot_clustering, alt_plot, plot_weight_dist
from pandas import DataFrame as df
from utils.calc import ArrayLike, BaseEstimator, stitch_mats
from sklearn.utils import safe_mask
import tensorly.decomposition as td
import tensorly as tl
from scipy.linalg import eig


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
            cluster_labels = np.argmax(w, 1)
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


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from scipy.linalg import svd
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda) ** 3 - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol: break
    return dot(Phi, R)


def estimate(x: ArrayLike, estimator: BaseEstimator, param_grid: dict, splits: int = 5):
    cv = [(slice(None), slice(None))]
    cv_ts = ms.TimeSeriesSplit(n_splits=splits)
    # estimator = LatentDirichletAllocation(max_iter=10000, learning_method="batch", evaluate_every=2)
    # estimator = AgglomerativeClustering()
    # estimator = KernelKMeans(n_init=10, verbose=2, max_iter=100)
    scoring = {  # 'sil': create_scorer(silhouette_score),
        'calinski': create_scorer(calinski_harabasz_score),
        # 'hom': create_scorer(homogeneity_score), 'comp': create_scorer(completeness_score),
        # 'v': create_scorer(v_measure_score),
        # 'fowlkes': create_scorer(fowlkes_mallows_score)
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

    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    SM, AUD, PROD = group_elecs(all_sigA, sig_chans)
    # %%
    cond = 'LSwords'
    # resp = all_sigA[cond]["Response"]
    # respz = all_sigZ[cond]["Response"]
    aud = all_sigA[cond]["AuditorywDelay"]
    audz = all_sigZ[cond]["AuditorywDelay"]
    # part = all_sigA[cond]["StartPart"]
    # partz = all_sigZ[cond]["StartPart"]
    go = all_sigA[cond]["DelaywGo"]
    goz = all_sigZ[cond]["DelaywGo"]
    # idx = AUD+SM+PROD
    x = np.unique(np.concatenate(list(sig_chans[cond].values())))
    x.sort()
    idx = SM#x.tolist()
    stitched = stitch_mats([aud[idx, :175], go[idx, :]], [0], axis=1)
    stitchedz = stitch_mats([audz[idx, :175], goz[idx, :]], [0], axis=1)
    stitchedz = stitchedz[np.where(np.any(np.abs(stitched) == 1, axis=1))[0]]
    stitched = stitched[np.where(np.any(np.abs(stitched) == 1, axis=1))[0]]
    stitchedw = np.multiply(stitchedz,stitched)
    cov = np.dot(stitchedz.T, stitchedz)
    eigen = eig(cov)
    vmax = varimax(eigen[1])
    # %%

    import nimfa

    bmf = nimfa.Bmf(stitched, seed="nndsvd", rank=4, max_iter=1000, lambda_w=1.01, lambda_h=1.01)
    bmf_fit = bmf()
    n = bmf.estimate_rank([2,3,4,5,6,7,8],n_run=100)
    from MEPONMF.onmf_DA import DA
    from MEPONMF.onmf_DA import ONMF_DA
    # k = 10
    # param = dict(tol=1e-8, alpha=1.002,
    #            purturb=0.5, verbos=1, normalize=False)
    # W, H, model = ONMF_DA.func(stitched, k=k, **param, auto_weighting=False)
    # model.plot_criticals(log=True)
    # plt.show()
    # k = model.return_true_number()
    # W, H, model2 = ONMF_DA.func(stitched, k=k, **param, auto_weighting=True)
    # model2 = DA(**param,K=k, max_iter=1000)
    # model2.fit(stitched,Px='auto')
    # Y,P = model2.cluster()
    # model2.plot_criticals(log=True)
    # plt.show()
    # plot_weight_dist(stitchedz, Y)
    # model2.pie_chart()

    # for k in np.array(range(7))+1:
    #     W, H, model = ONMF_DA.func(stitched, k, alpha=1.05, purturb=0.01, tol=1e-7)
    #     # model = DA(k, tol=1e-4, max_iter=1000, alpha=1.05,
    #     #             purturb=0.01, beta_final=None, verbos=0, normalize=False)
    #     # model.fit(stitched, Px='auto')
    #     # y, P = model.cluster()
    #     cost.append(model.return_cost())
    # x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched))
    # gridsearch = estimate(x, NMF(max_iter=100000, tol=1e-8), 3)
    # estimator = gridsearch.best_estimator_
    # estimator = NMF(max_iter=100000,init='nndsvda',alpha_W=0.01,
    #                              alpha_H=0.5, verbose=2)
    # estimator = FactorAnalysis(max_iter=100000,copy=True)
    # test = np.linspace(0, 1, 5)
    # # param_grid = {'n_components': [3], 'init': ['nndsvda'],
    # #               'solver': ['mu'], 'beta_loss': [2, 1, 0.5], 'l1_ratio': test,
    # #               'alpha_W': [0], 'alpha_H': test}
    # param_grid = {'n_components': [4],'rotation' : ['varimax', 'quartimax']}
    # gridsearch = estimate(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)), estimator,
    #                       param_grid, 5)
    # estimator = gridsearch.best_estimator_
    # estimator.n_components = 4
    # y = estimator.fit_transform(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)))
    # res = df(gridsearch.cv_results_)
    # decomp = td.non_negative_parafac
    # tens = td.CP_NN_HALS(3, n_iter_max=10000, init='random', exact=True, tol=1e-7)
    # tens.mask = tl.tensor(stitched)
    # tens.fit(tl.tensor(stitchedz))
    # y = tens.decomposition_.factors[0]
    #k = 4
    #W, H, model = ONMF_DA.func(stitched, k, alpha=1.05, purturb=0.01, tol=1e-4)
    #plot_weight_dist(stitched, W)

    #
    # gridsearch.scorer_ = gridsearch.scoring = {}
    # np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
    # plt.plot(decomp_sigs)
    # plt.savefig('data/decomp.png')
