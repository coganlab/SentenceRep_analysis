import numpy as np
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape, silhouette_score
import sklearn.decomposition
from sklearn.decomposition import NMF, LatentDirichletAllocation
from tslearn.utils import to_sklearn_dataset
from tslearn.neighbors import KNeighborsTimeSeries as NearestNeighbors
from tslearn.metrics import gamma_soft_dtw, dtw, gak
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from sklearn.metrics import make_scorer, calinski_harabasz_score
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mat_load import get_sigs, load_all, group_elecs
from calc import calc_score, get_elbow, dist, mat_err
from pandas import DataFrame as df
from typing import Union, Any, Iterable


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


def sk_clustering(x, k, metric='euclidean'):
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


def plot_dist(mat: iter, label: Union[str, int, float] = None,
              color: Union[str, list[int]] = None) -> plt.Axes:
    mean, std = dist(mat)
    tscale = range(len(mean))
    plt.errorbar(tscale, mean, yerr=std, label=label, color=color)
    return plt.gca()


def clustering_subplots(data, label,  sig_titles, colors, weighted):
    fig, ax = plt.subplots()
    if weighted:
        group = range(min(np.shape(label)))
    else:
        group = np.unique(label)
    if sig_titles is None:
        sig_titles = [sig_titles] * len(group)
    if colors is None:
        colors = [colors] * len(group)
    for i, stitle, color in zip(group, sig_titles, colors):
        if not weighted:
            w_sigs = data[label == i]
        else:
            try:
                w_sigs = np.array([label[i][j] * dat for j, dat in enumerate(data.T)])
            except (ValueError, IndexError) as e:
                w_sigs = np.array([label.T[i][j] * dat for j, dat in enumerate(data)])
        ax = plot_dist(w_sigs, stitle, color)
    return fig, ax


def plot_clustering(data: np.ndarray, label: np.ndarray,
                    sig_titles: Iterable[str] = None, weighted: bool = False,
                    colors: Iterable[Union[str, list[Union[int, float]]]] = None):

    fig, ax = clustering_subplots(data, label,  sig_titles, colors, weighted)
    # the x coords of this transformation are data, and the
    # y coord are axes
    trans = ax.get_xaxis_transform()
    ax.text(50, 0.8, 'Stim onset', rotation=270, transform=trans)
    ax.axvline(175)
    ax.axvline(50, linestyle='--')
    ax.axvline(225, linestyle='--')
    # ax.axhline(0, linestyle='--', color='black')
    ax.text(225, 0.87, 'Go cue', rotation=270, transform=trans)
    ax.text(160, 0.6, 'Delay', transform=trans)
    # ax.legend(loc="best")
    ax.axvspan(150, 200, color=(0.5, 0.5, 0.5, 0.15))
    ax.set_xticks([0, 50, 100, 150, 200, 225, 250, 300, 350],
                  ['-0.5', '0', '0.5', '1', '-0.25', '0', '0.25', '0.75', '1.25'])
    ax.set_xlabel('Time from stimuli or go cue (seconds)')
    # ax.set_ylabel('Z score')
    ax.set_ylabel('Significance of activity (range [0-1])')
    ax.set_xlim(0, 350)
    ylims = ax.get_ybound()
    ax.set_ybound(min(0,ylims[0]),ylims[1])
    # plt.title(title)
    plt.show()
    return fig, ax


def plot_clustering_resp(data: np.ndarray, label: np.ndarray,
                    sig_titles: Iterable[str] = None, weighted: bool = False,
                    colors: Iterable[Union[str, list[Union[int, float]]]] = None, ybounds=None):
    fig, ax = clustering_subplots(data, label,  sig_titles, colors, weighted)
    trans = ax.get_xaxis_transform()
    ax.text(100, 0.9, 'onset', rotation=270, transform=trans)
    ax.axvline(100, linestyle='--')
    # ax.axvline(50, linestyle='--')
    # ax.axvline(225, linestyle='--')
    # ax.text(225, 0.87, 'go cue', rotation=270, transform=trans)
    # ax.text(152, 0.6, 'transition',  transform=trans)
    ax.legend(loc="best")
    # ax.axvspan(150,200,color=(0.5,0.5,0.5,0.15))
    ax.set_ybound(ybounds)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200],
                  ['-1', '-0.5', '0', '0.5', '1'])
    ax.set_xlabel('Time from response (seconds)')
    # ax.set_ylabel('Significance of activity (range [0-1])')
    ax.set_xlim(0, 200)
    # if title is not None:
    #     plt.title(title)
    plt.show()


def do_decomp(data, clusters=8, repetitions=10, mod=NMF(init='random', max_iter=10000, verbose=2)):
    data = data - np.min(data)
    errs = np.ndarray([repetitions, clusters])
    for k in np.array(range(clusters)):
        mod.set_params(n_components=k + 1)
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
        dat[dat >= 1] = sigZ[group][dat >= 1]
        alt_plot(dat, labels)
    return scores, models


def estimate():
    cv = [(slice(None), slice(None))]
    cv_ts = ms.TimeSeriesSplit(n_splits=2)
    estimator = NMF(max_iter=100000)
    # estimator = LatentDirichletAllocation(max_iter=10000, learning_method="batch", evaluate_every=2)
    # estimator = AgglomerativeClustering()
    # estimator = KernelKMeans(n_init=10, verbose=2, max_iter=100)
    test = np.linspace(0, 1, 5)
    param_dict_sil = {'n_components': [2, 3, 4], 'init': ['random', 'nndsvd', 'nndsvda'],
                      'solver': ['mu'], 'beta_loss': np.linspace(-0.5, 3, 8),
                      'l1_ratio': test}
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
    gs.fit(df(x))
    keys = list(gs.best_estimator_.__dict__.keys())
    thing = keys[comp == keys]
    if gs.best_estimator_.__dict__[thing] == 2:
        gs.param_grid[comp] = [1, 2, 3, 4]
        gs.scoring = create_scorer(calinski_harabasz_score)
        gs.fit(df(x))
    winner = gs.best_estimator_
    # keys = list(gs.best_estimator_.__dict__.keys())
    # thing = keys[comp == keys]

    gs.estimator = winner['SM']
    gs.scoring = {'sil': create_scorer(silhouette_score), 'calinski': create_scorer(calinski_harabasz_score)}
    gs.refit = 'calinski'
    gs.param_grid = {comp: [i + 2 for i in range(winner.__dict__[thing] + 2)]}


if __name__ == "__main__":
    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    SM, AUD, PROD = group_elecs(all_sigA, sig_chans)
    npSM = np.array(SM)
    cond = 'LSwords'
    SMresp = npSM[npSM < len(all_sigA[cond]['Response'])]
    resp = all_sigA[cond]['Response'][SMresp, :]
    sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans, cond)
    winners, results, w_sav = np.load('data/nmf.npy', allow_pickle=True)
    SMrespw = w_sav['SM'][npSM < len(all_sigA['LSwords']['Response'])]
    ones = np.ones([244, 1])
    x = np.array(sigZ['AUD'])
    labels = np.ones([np.shape(sigZ['AUD'])[0]])
    x = np.vstack([x,np.array(sigZ['SM'])])
    labels = np.concatenate([labels, np.ones([np.shape(sigZ['SM'])[0]]) * 2])
    x = np.vstack([x, np.array(sigZ['PROD'])])
    labels = np.concatenate([labels, np.ones([np.shape(sigZ['PROD'])[0]]) * 3])
    colors = [[0,0,0],[0.6,0.3,0],[.9,.9,0],[1,0.5,0]]
    plot_clustering(sigA['SM'], np.ones([244, 1]), None, True,[[1,0,0]])
                    # ['Working Memory','Visual','Early Prod','Late Prod'], True,
                    # [[0,1,0],[1,0,0],[0,0,1]])
    ax = plt.gca()
    ylims = list(ax.get_ybound())
    ylims[0] = min(0,ylims[0])
    # plt.title('Listen-speak')
    plot_clustering_resp(resp, np.ones([180, 1]), ['SM'], True,
                         [[1,0,0]],ylims)

    # w={}
    # name = {'SM':'Sensory-Motor','AUD':'Auditory','PROD':'Production'}
    # for group, x in sigA.items():
    #     # w[group] = winners[group].fit_transform(x)
    #     plot_clustering(x, w_sav[group], True, name[group], True)
    # plt.savefig(group + "fig.svg",dpi=300,format='svg')
