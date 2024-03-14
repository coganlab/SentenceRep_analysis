import os.path
from grouping import GroupData
import numpy as np
from sklearn.cluster import OPTICS
from tslearn.metrics import soft_dtw
from tslearn.clustering import TimeSeriesKMeans
from joblib import Parallel, delayed
from collections.abc import Sequence
import matplotlib.pyplot
from analysis.decomposition import explained_variance, silhouette, calinski_harabasz, davies_bouldin


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
sub_ave = GroupData.from_intermediates("SentenceRep", fpath, folder='ave')

conds_aud = ['aud_ls', 'aud_lm', 'aud_jl']
conds_go = ['go_ls', 'go_lm', 'go_jl']
aud = sub.array['zscore', conds_aud]
aud.labels[0] = aud.labels[0].replace("aud_", "")
go = sub.array['zscore', conds_go]
go.labels[0] = go.labels[0].replace("go_", "")
aud_go = aud[..., :175].concatenate(go, -1)

ls = np.moveaxis(np.nanmean(aud_go.combine((0, 1)), axis=2), 0, -1)
# ls_sm = np.nanmean(ls[list(sub.SM),].dropna(), axis=2)
# scores = get_k(ls, tsk_func, range(1, 10), n_jobs=6)
estimator = TimeSeriesKMeans(n_clusters=3,
                             metric="dtw",
                             tol=1,
                             n_jobs=7,
                             verbose=40)
estimator.fit(ls)
