import numpy as np
from scipy import linalg
import os
from ieeg.calc.mat import LabeledArray
from analysis.grouping import GroupData
import scipy.stats as st
from tqdm import tqdm

n_samples, n_features, rank = 500, 25, 5
sigma = 1.0
rng = np.random.RandomState()
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.0
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
## setup training data
aud_slice = slice(0, 175)
pval = np.where(sub.p_vals > 0.9999, 0.9999, sub.p_vals)

# pval[pval<0.0001] = 0.0001
zscores = LabeledArray(st.norm.ppf(1 - pval), sub.p_vals.labels)
powers = np.nanmean(sub['zscore'].array, axis=(-4, -2))
X_hetero = np.hstack([zscores['aud_ls', :, aud_slice],
                    # zscores['aud_lm', :, aud_slice],
                    # zscores['aud_jl', :, aud_slice],
                    zscores['go_ls'],
                    # zscores['go_lm'],
                    # zscores['go_jl'],
                    # zscores['resp']
                     ])[list(sub.SM)]
n_features = X_hetero.shape[1]
import matplotlib.pyplot as plt

from sklearn.covariance import OAS, ShrunkCovariance
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score

n_components = np.arange(0, n_features//8, 5)  # options for n_components


def compute_scores(X):
    pca = PCA(svd_solver="full")
    fa = FactorAnalysis(rotation="varimax")

    pca_scores, fa_scores = [], []
    for n in tqdm(n_components):
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(OAS(), X))


X, title = X_hetero, "Heteroscedastic Noise"
pca_scores, fa_scores = compute_scores(X)
n_components_pca = n_components[np.argmax(pca_scores)]
n_components_fa = n_components[np.argmax(fa_scores)]

# pca = PCA(svd_solver="full", n_components="mle")
# pca.fit(X)
# n_components_pca_mle = pca.n_components_

print("best n_components by PCA CV = %d" % n_components_pca)
print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
# print("best n_components by PCA MLE = %d" % n_components_pca_mle)

plt.figure()
plt.plot(n_components, pca_scores, "b", label="PCA scores")
plt.plot(n_components, fa_scores, "r", label="FA scores")
plt.axvline(rank, color="g", label="TRUTH: %d" % rank, linestyle="-")
plt.axvline(
    n_components_pca,
    color="b",
    label="PCA CV: %d" % n_components_pca,
    linestyle="--",
)
plt.axvline(
    n_components_fa,
    color="r",
    label="FactorAnalysis CV: %d" % n_components_fa,
    linestyle="--",
)
# plt.axvline(
#     n_components_pca_mle,
#     color="k",
#     label="PCA MLE: %d" % n_components_pca_mle,
#     linestyle="--",
# )

# compare with other covariance estimators
plt.axhline(
    shrunk_cov_score(X),
    color="violet",
    label="Shrunk Covariance MLE",
    linestyle="-.",
)
# plt.axhline(
#     lw_score(X),
#     color="orange",
#     label="LedoitWolf MLE" % n_components_pca_mle,
#     linestyle="-.",
# )

plt.xlabel("nb of components")
plt.ylabel("CV scores")
plt.legend(loc="lower right")
plt.title(title)

plt.show()