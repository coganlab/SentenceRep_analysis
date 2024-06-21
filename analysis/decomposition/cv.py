from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.decomposition import NMF
from analysis.grouping import GroupData
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform


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


def calinski_harabasz(X: np.ndarray, H: np.ndarray):
    k = H.shape[0]
    # create groups
    groups = np.zeros((k,) + X.shape)
    for i in range(k):
        groups[i] = X * H[i]

    return avg_group_distances_weighted(groups.transpose(0, 2, 1), H)


def avg_group_distances_weighted(matrices, weights):
    # Initialize sums and counts for within and between group distances
    within_sum = 0
    within_count = 0
    between_sum = 0
    between_count = 0

    # Calculate within-group distances
    for matrix, weight in zip(matrices, weights):
        distances = squareform(pdist(matrix, 'euclidean'))
        weight_matrix = np.outer(weight, weight)  # Create a 2D weight matrix
        weighted_distances = distances * weight_matrix
        within_sum += np.sum(weighted_distances)
        within_count += distances.size

    # Calculate between-group distances
    for i in range(len(matrices)):
        for j in range(i+1, len(matrices)):
            distances = squareform(pdist(np.concatenate((matrices[i], matrices[j])), 'euclidean'))
            weight = np.concatenate((weights[i], weights[j]))
            weight_matrix = np.outer(weight, weight)  # Create a 2D weight matrix
            weighted_distances = distances * weight_matrix
            between_sum += np.sum(weighted_distances)
            between_count += distances.size

    # Calculate average within and between group distances
    avg_within = within_sum / within_count
    avg_between = between_sum / between_count

    # Return the ratio of average between-group to within-group distance
    return avg_between / avg_within


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
        calinski_harabasz=calinski_harabasz(X, H),
        sparsity=sparsity(W, H),
        orthogonality=orthogonality(W),
        reconstruction_error=reconstruction_error(X, W, H),
        explained_variance=evar(X, W, H))

nmf = NMF(init='random', max_iter=100000, solver='cd')
grid = GridSearchCV(nmf, param_grid, cv=10, n_jobs=-2, verbose=10,
                    scoring=scorer, refit='calinski_harabasz')

grid.fit(trainz.T)

# %% plot


