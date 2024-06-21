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
    k = W.shape[1]
    # create groups
    groups = np.zeros((k,) + X.T.shape)
    for i in range(k):
        groups[i] = X.T * W[:, i].T

    # within group scatter
    B = np.zeros(X.shape)
    for i in range(k):
        B += np.linalg.norm(groups[i] - np.mean(groups[i], axis=0), axis=1)
    B = np.sum(B)

    # between group scatter
    grp = np.sum(np.linalg.norm(W, axis=0))

    return B / grp


def calinski_scorer(est, X, y=None):
    calinski_harabasz(X, est.transform(X))


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
              'alpha_W': [0, 0.1, 1],
                'alpha_H': [0, 0.1, 1],
                'l1_ratio': np.linspace(0, 1, 4)}

def scorer(est, X, y=None):
    W = est.transform(X)
    H = est.components_
    return dict(
        calinski_harabasz=calinski_harabasz(X, W),
        sparsity=sparsity(W, H),
        orthogonality=orthogonality(W),
        reconstruction_error=reconstruction_error(X, W, H),
        explained_variance=evar(X, W, H))

scoring = {'calinski_harabasz': calinski_scorer,
              'sparsity' : sparsity_scorer,
              'orthogonality': ortho_scorer,
              'reconstruction_error': recon_scorer,
              'explaned_varience': evar_scorer}
nmf = NMF(init='random', max_iter=100000, solver='cd')
grid = GridSearchCV(nmf, param_grid, cv=10, n_jobs=1, verbose=10,
                    scoring=scorer, refit='calinski_harabasz')

grid.fit(trainz)

# %% plot


