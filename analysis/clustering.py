import os
import numpy as np
from analysis.grouping import GroupData
from sktime.clustering.dbscan import TimeSeriesDBSCAN
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.cluster import OPTICS
from sktime.dists_kernels import FlatDist, ScipyDist, AggrDist
from ieeg.calc.mixup import mixupnd
from ieeg.viz import _qt_backend
from sktime.distances import dtw_distance
_qt_backend()
from scipy.sparse import csr_matrix, issparse

import matplotlib.pyplot as plt


fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath,
                                   folder='stats_opt', wide=True)
conds = {"resp": (-1, 1),
         "aud_ls": (-0.5, 1.5),
         "aud_lm": (-0.5, 1.5),
         "aud_jl": (-0.5, 1.5),
         "go_ls": (-0.5, 1.5),
         "go_lm": (-0.5, 1.5),
         "go_jl": (-0.5, 1.5)}
# %% setup training data
aud_slice = slice(0, 175)
stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                      sub.signif['aud_lm', :, aud_slice],
                      sub.signif['go_ls', :],
                      sub.signif['resp', :]])
zscores = np.nanmean(sub['zscore'].array, axis=(-4, -2))
# zscores = sub['zscore'].array.combine((1, 3))
# powers = np.nanmean(sub['power'].array, axis=(-4, -2))
sig = sub.signif

# trainz = np.hstack([zscores['aud_ls', ..., aud_slice],
#                    zscores['aud_lm', ..., aud_slice],
#                    zscores['go_ls'], zscores['resp']])
trainz = np.concatenate([zscores['aud_ls', ..., aud_slice],
                         zscores['aud_lm', ..., aud_slice],
                         zscores['go_ls'], zscores['resp']], axis=-1)
# trainz = np.nanmean(trainz, axis=1, keepdims=True)
# mixupnd(trainz, 1)
sparse_matrix = csr_matrix((trainz[stitched == 1], stitched.nonzero()))
# sparse_matrix.data -= np.min(sparse_matrix.data)

# %% DBSACAN

# eucl_dist = AggrDist(ScipyDist(), aggfunc_is_symm=True,
#                      aggfunc=np.nanmean)
# input_mat = csr_matrix((trainz[~np.isnan(trainz)], np.where(~np.isnan(trainz))))
# y = eucl_dist.transform(sparse_matrix[list(sub.SM)])
# plt.hist(y.flat)
# clust = TimeSeriesDBSCAN(eucl_dist, 10, n_jobs=-1)
# clust.fit(trainz[list(sub.SM)])
# clust.get_fitted_params()
# plt.hist(clust.labels_)
# plt.show()

# %% OPTICS
optics = OPTICS(n_jobs=-1, metric='euclidean')
optics.fit(sparse_matrix[list(sub.SM)])
plt.hist(optics.labels_)