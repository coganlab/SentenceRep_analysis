from analysis.decomposition.pick_k import calinski_harabasz, davies_bouldin, silhouette_score, \
    orthogonality, reconstruction_error, sparsity, evar
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.decomposition import NMF
from analysis.grouping import GroupData
import os
import numpy as np

def evar(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return 1 - np.linalg.norm(orig - W @ H) ** 2 / np.linalg.norm(orig) ** 2


def evar_scorer(est, X, y=None):
    return evar(X, est.transform(X), est.components_)


def reconstruction_error(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(orig - W @ H)


def recon_scorer(est, X, y=None):
    err = reconstruction_error(X, est.transform(X), est.components_)
    return -err


def orthogonality(W: np.ndarray) -> float:
    return np.linalg.norm(W.T @ W - np.eye(W.shape[1]))


def ortho_scorer(est, X, y=None):
    return orthogonality(est.transform(X))


def sparsity(W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(W, ord=1) + np.linalg.norm(H, ord=1)


def sparsity_scorer(est, X, y=None):
    return sparsity(est.transform(X), est.components_)


def calinski_harabasz(X: np.ndarray, W: np.ndarray):
    """ratio of the sum of between-cluster dispersion and of within-cluster dispersion"""



def calinski_scorer()

# %%
# Load the data
kwargs = dict(folder='stats')
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, **kwargs)

# %% get data
zscore = np.nanmean(sub['zscore'].array, axis=(-4, -2))
aud_slice = slice(0, 175)
met = zscore
trainz = np.hstack([met['aud_ls', :, aud_slice],
                    met['aud_lm', :, aud_slice],
                    # met['aud_jl', :, aud_slice],
                    met['go_ls'],
                    # met['go_lm'],
                    # met['go_jl'],
                    met['resp']])
# powers['resp']])
idx = sorted(list(sub.SM))
trainz = trainz[idx]
trainz -= np.min(trainz)
trainz /= np.max(trainz)

# %% set up cross-validation
param_grid = {'n_components': np.arange(1, 11),
              'solver': ['cd', 'mu'],
              'beta_loss': ['frobenius', 'kullback-leibler']}
scoring = {'calinski_harabasz': calinski_harabasz,
              'davies_bouldin': davies_bouldin,
              'silhouette_score': silhouette_score,
              'orthogonality': orthogonality,
              'reconstruction_error': lambda est: est.reconstruction_err_,
              'sparsity': sparsity,
              'explaned_varience': evar}
nmf = NMF(init='random', max_iter=100000)
grid = GridSearchCV(nmf, param_grid, cv=5, n_jobs=1, verbose=10,
                    scoring=recon_scorer, refit=True)

grid.fit(trainz)

