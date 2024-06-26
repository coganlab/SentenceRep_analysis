from sklearn.model_selection import GridSearchCV
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import NMF
from analysis.grouping import GroupData
import os
import numpy as np
from ieeg.viz.ensemble import plot_weight_dist


def evar(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return 1 - np.linalg.norm(orig - W @ H) ** 2 / np.linalg.norm(orig) ** 2


def reconstruction_error(orig: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(orig - W @ H)


def orthogonality(W: np.ndarray) -> float:
    return np.linalg.norm(W.T @ W - np.eye(W.shape[1]))


def sparsity(W: np.ndarray, H: np.ndarray) -> float:
    return np.linalg.norm(W, ord=1) + np.linalg.norm(H, ord=1)


def probabilistic_calinski_harabasz(X, probabilities):
    """
    Calculates the Calinski-Harabasz score using probability percentages.
    Args:
        X (array-like): Data points (samples) with shape (n_samples, n_features).
        probabilities (array-like): Probability percentages for each data point, shape (n_samples, n_clusters).
    Returns:
        float: Calinski-Harabasz score.
    """
    # Compute cluster centroids based on probabilities
    centroids = np.dot(probabilities.T, X) / np.sum(probabilities, axis=0)[:, np.newaxis]
    # Calculate within-cluster dispersion (W)
    W = np.sum(probabilities * pairwise_distances(X, centroids) ** 2)
    # Calculate between-cluster dispersion (B)
    global_centroid = np.mean(X, axis=0)
    B = np.sum(np.sum(probabilities, axis=0) * pairwise_distances(centroids, global_centroid.reshape(1, -1)) ** 2)
    # Compute the CH score
    n_samples, n_clusters = probabilities.shape
    CH_score = (B / (n_clusters - 1)) / (W / (n_samples - n_clusters))
    return CH_score
# Example usage:
# Replace 'X' and 'probabilities' with your actual data and probabilities
# X, probabilities = load_your_data()
# ch_score = calinski_harabasz_score_with_probabilities(X, probabilities)
# print(f"Calinski-Harabasz score: {ch_score:.2f}")


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
# trainz /= np.max(trainz)

# %% set up cross-validation
param_grid = {'n_components': np.arange(2, 11),
              'solver': ['cd'], 'tol': [1e-4],
              'alpha_W': [0, 0.5, 1], 'alpha_H': [0, 0.5, 1],
              'l1_ratio': [0, 0.5, 1], 'shuffle': [True]}

def safe_score(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return np.nan

def scorer(est, X, y=None):
    try:
        W = est.transform(X)
        H = est.components_
        probabilities = (H / np.sum(H, axis=0)).T
    except Exception:
        return dict(
            calinski_harabasz=np.nan,
            sparsity=np.nan,
            orthogonality=np.nan,
            reconstruction_error=np.nan,
            explained_variance=np.nan)
    return dict(
        calinski_harabasz=safe_score(probabilistic_calinski_harabasz, X.T, probabilities),
        sparsity=safe_score(sparsity, W, H),
        orthogonality=safe_score(orthogonality, H),
        reconstruction_error=safe_score(reconstruction_error, X, W, H),
        explained_variance=safe_score(evar, X, W, H))

nmf = NMF(init='random', max_iter=100000)
grid = GridSearchCV(nmf, param_grid, cv=10, n_jobs=1, verbose=10,
                    scoring=scorer, refit='calinski_harabasz')
# , error_score=err)

grid.fit(trainz.T)

# %% get results
results = dict()
stds = dict()
for k, v in grid.cv_results_.items():

    if 'mean_test_' in k:
        metric = k.split('_')[-1]
        results[metric] = v
    elif 'std_test_' in k:
        metric = k.split('_')[-1]
        stds[metric] = v
# %% plot

import matplotlib.pyplot as plt

score = 'harabasz'
plt.plot(np.arange(2, 11), results[score])
plt.fill_between(np.arange(2, 11), results[score] - stds[score],
                    results[score] + stds[score], alpha=0.5)
plt.xlabel('Number of components')
plt.ylabel(f'{score} score')

# %% plot estimated components
best = grid.best_estimator_
W = best.transform(trainz.T)
H = best.components_
plt.figure()

conds = {"resp": (-1, 1),
         "aud_ls": (-0.5, 1.5),
         "aud_lm": (-0.5, 1.5),
         "aud_jl": (-0.5, 1.5),
         "go_ls": (-0.5, 1.5),
         "go_lm": (-0.5, 1.5),
         "go_jl": (-0.5, 1.5)}
metric = zscore
labeled = [['Instructional', 'c', 1],
           ['Motor', 'm', 2],
           ['Feedback', 'k', 0],]
# ['Working Memory', 'orange', 4],]
# ['Auditory', 'g', 3]]
pred = np.argmax(H, axis=0)
groups = [[idx[i] for i in np.where(pred == j)[0]]
          for j in range(H.shape[0])]
colors, labels = zip(
    *((c[1], c[0]) for c in sorted(labeled, key=lambda x: x[2])))
for i, g in enumerate(groups):
    labeled[i][2] = groups[labeled[i][2]]
cond = 'resp'
if cond.startswith("aud"):
    title = "Stimulus Onset"
    times = slice(None)
    ticks = [-0.5, 0, 0.5, 1, 1.5]
    legend = True
elif cond.startswith("go"):
    title = "Go Cue"
    times = slice(None)
    ticks = [-0.5, 0, 0.5, 1, 1.5]
    legend = False
else:
    title = "Response"
    times = slice(None)
    ticks = [-1, -0.5, 0, 0.5, 1]
    legend = False
plot = metric[cond, idx, times].__array__().copy()
# plot[sig[cond, sub.SM, times].__array__() == 0] = np.nan
ax = plot_weight_dist(plot, pred, times=conds[cond], colors=colors,
                           sig_titles=labels)
plt.xticks(ticks)
plt.rcParams.update({'font.size': 14})
# plt.ylim(0, 1.5)
# sort by order
labeled = sorted(labeled, key=lambda x: np.median(
    np.argmax(metric[cond, x[2]].__array__(), axis=1)))
ylim = ax.get_ylim()

for i, (label, color, group) in enumerate(labeled):
    points = metric[cond, group, times].__array__()
    # points[sig[cond, group, times].__array__() == 0] = 0
    peaks = np.max(points, axis=1)
    peak_locs = np.argmax(points, axis=1)
    tscale = np.linspace(conds[cond][0], conds[cond][1], metric.shape[-1])
    # ax.scatter(tscale[peak_locs], peaks, color=colors[i], label=labels[i])
    # plot horizontal box plot of peak locations
    spread = (ylim[1] - ylim[0])
    width = spread / 16
    pos = [width / 2 + i * width + spread * 3 / 4]
    # boxplot_2d(tscale[peak_locs], peaks, ax)
    bplot = ax.boxplot(tscale[peak_locs], manage_ticks=False, widths=width,
                       positions=pos, boxprops=dict(facecolor=color,
                                                    fill=True, alpha=0.5),
                       vert=False, showfliers=False, whis=[15, 85],
                       patch_artist=True, medianprops=dict(color='k'))

trans = ax.get_xaxis_transform()
# ax.text(50, 0.8, title, rotation=270, transform=trans)
plt.ylabel("Z-scored HG power")
plt.xlabel(f"Time from {title} (s)")
plt.xlim(*conds[cond])
if legend:
    plt.legend()
plt.tight_layout()

# %% plot on brain
pred = np.argmax(H, axis=0)
groups = [[sub.keys['channel'][idx[i]] for i in np.where(pred == j)[0]]
          for j in range(W.shape[1])]
fig1 = sub.plot_groups_on_average(groups,
                                  ['k', 'c', 'm', 'orange'],
                                  hemi='lh',
                                  rm_wm=False)