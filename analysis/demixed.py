import os.path
from grouping import GroupData
import numpy as np
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.dists_kernels import FlatDist, LuckyDtwDist
from joblib import Parallel, delayed
from collections.abc import Sequence
import matplotlib.pyplot
from analysis.pick_k import explained_variance, silhouette, calinski_harabasz, davies_bouldin
from analysis.utils.plotting import plot_weight_dist
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


def tsk_func(X, n_clusters, metric="softdtw"):
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, n_jobs=1, verbose=10)
    km.fit(X)
    Y = km.predict(X)
    H = km.cluster_centers_
    return Y, H


def get_k(X: np.ndarray, estimator, k_test: Sequence[int] = range(1, 10),
          metric: callable = explained_variance, n_jobs: int = -3,
          reps: int = 10) -> tuple[matplotlib.pyplot.Axes, np.ndarray]:

    def _repeated_estim(k: int, estimate=estimator) -> np.ndarray[float]:
        est = []
        for i in range(reps):
            Y, H = estimate(X, k)
            est.append(metric(X, Y, H))
        return np.array(est)

    par_gen = Parallel(n_jobs=n_jobs, verbose=10, return_as='generator')(
        delayed(_repeated_estim)(k) for k in k_test)
    est = np.zeros((reps, len(k_test), 4))
    for i, o in enumerate(par_gen):
        est[:, i, :] = o[...]

    return est

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt', wide=False)
# sub_ave = GroupData.from_intermediates("SentenceRep", fpath, folder='ave')

conds_aud = ['aud_ls', 'aud_lm', 'aud_jl']
conds_go = ['go_ls', 'go_lm', 'go_jl']
aud = sub.array['zscore', conds_aud]
audp = sub.p_vals[conds_aud,]
aud.labels[0] = aud.labels[0].replace("aud_", "")
audp.labels[0] = audp.labels[0].replace("aud_", "")
go = sub.array['zscore', conds_go]
gop = sub.p_vals[conds_go,]
go.labels[0] = go.labels[0].replace("go_", "")
gop.labels[0] = gop.labels[0].replace("go_", "")
aud_go = aud[..., :175].concatenate(go, -1)
aud_go_p = audp[..., :175].concatenate(gop, -1)

ls = np.moveaxis(np.nanmean(aud_go, axis=(1, 3)), 0, 1)
lsp = np.moveaxis(aud_go_p, 0, 1)
ls_sm = ls[list(sub.SM),].dropna()
lsp_sm = lsp[list(sub.SM),]
# scores = get_k(ls, tsk_func, range(1, 10), n_jobs=6)
euc_dist = FlatDist(LuckyDtwDist())
estimator = TimeSeriesKMeans(n_clusters=4,
                             tol=1e-14,
                             metric="euclidean",
                             verbose=True)
estimator.fit(lsp_sm[:,'ls'].__array__())

#
avg_ls = ls_sm[:,'ls'].__array__()
fig, ax = plot_weight_dist(avg_ls, estimator.labels_)
bins = np.bincount(estimator.labels_)
plot_cluster_algorithm(estimator, ls_sm[:,'ls', None].__array__(), len(bins))
