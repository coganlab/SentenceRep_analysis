# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification
from ieeg.viz.utils import plot_dist
from ieeg.calc.reshape import smote as do_smote
from ieeg.calc.mat import LabeledArray
from joblib import Parallel, delayed
from copy import deepcopy


class Decoder(PcaLdaClassification):

    def __init__(self, *args, cv=RepeatedStratifiedKFold,
                 categories: dict = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cv = cv
        self.categories = categories

    def cv_cm(self, x_data: LabeledArray, kfolds: int = 5,
              repeats: int = 10, smote: bool = True, normalize: str = 'true',
              verbose: bool = True):
        rep = 0
        n_cats = len(self.categories)
        mats = np.zeros((repeats, kfolds, n_cats, n_cats))
        data = x_data.combine((x_data.ndim - 2, x_data.ndim - 1))
        for f, (x_train, x_test, y_train, y_test) in enumerate(
                self.splits(data, kfolds, 1, smote)):
            fold = f + 1 - rep * kfolds
            if verbose:
                print("Fold {} of {}".format(fold, kfolds))
            self.fit(x_train, y_train)
            pred = self.predict(x_test)
            mats[rep, fold-1] = confusion_matrix(y_test, pred)
            if f - rep * kfolds == kfolds - 1:
                rep += 1
        mats = np.mean(np.sum(mats, axis=1), axis=0)
        if normalize == 'true':
            return mats / np.sum(mats, axis=1)
        elif normalize == 'pred':
            return mats / np.sum(mats, axis=0)
        elif normalize == 'all':
            return mats / np.sum(mats)
        else:
            return mats

    def sliding_window(self, x_data: LabeledArray, window_size: int = 20,
                       axis: int = -1,kfolds: int = 5, repeats: int = 10,
                       smote: bool = True, normalize: str = 'true',
                       n_jobs: int = -3) -> np.ndarray:

        # make windowing generator
        axis = x_data.ndim + axis if axis < 0 else axis
        slices = (slice(start, start + window_size)
                  for start in range(0, x_data.shape[axis] - window_size))
        idxs = ([slice(None) if i != axis else sl for i in range(x_data.ndim)]
                for sl in slices)

        # initialize output array
        n_cats = len(self.categories)
        out = np.zeros((x_data.shape[axis] - window_size + 1, n_cats, n_cats))

        # Use joblib to parallelize the computation
        gen = Parallel(n_jobs=n_jobs, return_as='generator', verbose=20)(
            delayed(self.cv_cm)(x_data[idx], kfolds, repeats, smote, normalize,
                                False) for idx in idxs)
        for i, mat in enumerate(gen):
            out[i] = mat

        return out

    def splits(self, x_data: LabeledArray, folds: int = 5, obs_axs: int = -2,
               smote: bool = True):

        obs_axs = list(range(x_data.ndim))[obs_axs]
        non_trial_dims = tuple(i for i in range(x_data.ndim + 1) if i != obs_axs)
        delim = x_data.labels[0].delimiter

        # max_trials = data.shape[obs_axs] // folds * folds
        f_idx = np.random.choice(np.arange(x_data.shape[obs_axs]),
                                 (x_data.shape[obs_axs] // folds, folds),
                                 False)
        idx = list(slice(None) if i != obs_axs else f_idx for i in
               range(x_data.ndim))
        i = 0
        while np.sum(np.any(np.isnan(x_data.__array__()[tuple(idx)]),
                            axis=non_trial_dims) == False) < 2:
            f_idx = np.random.choice(np.arange(x_data.shape[obs_axs]),
                                     (x_data.shape[obs_axs]//folds, folds),
                                     False)
            idx[obs_axs] = f_idx
            i += 1
            if i > 100000:
                raise RecursionError("Could not find a valid split")

        for j in range(folds):
            idx[obs_axs] = f_idx[..., j]
            X_test = deepcopy(x_data[idx])
            not_j = [i for i in range(folds) if i != j]
            idx[obs_axs] = f_idx[..., tuple(not_j)].flatten()
            X_train = deepcopy(x_data[idx])
            if smote:
                do_smote(X_test.__array__(), obs_axs, False)
                do_smote(X_train.__array__(), obs_axs, False)
            X_test = X_test.combine((0, 1))
            X_train = X_train.combine((0, 1))
            y_test = np.array(
                [self.categories[k.split(delim)[0]] for k in X_test.labels[0]])
            y_train = np.array(
                [self.categories[k.split(delim)[0]] for k in X_train.labels[0]])
            yield X_train.__array__(), X_test.__array__(), y_train, y_test



# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')


# # %%
# scores = []
# plt_labels = []
# modes = [3,4,5,6,7,8,9,10,11,12,13,14,15,"smote-17"]
# for i in modes:
#     # Arrange
#     conds = ['aud_ls', 'aud_lm', 'aud_jl']
#     idx = sub.PROD
#     reduced = sub[:, conds, :, idx, :, 50:125]
#     reduced.array = reduced.array.dropna()
#     if i != "smote-17":
#         reduced = reduced.nan_common_denom(True, i, True)
#     else:
#         reduced.smotify_trials()
#     comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'trial'))
#     # concatenate channels across time as feature vector
#     x_data = comb['power'].array.combine((0, 2)).dropna()
#     cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
#     labels = np.array([cats[k.split('-')[1]] for k in x_data.labels[0]])
#
#     # Decoding
#     kfolds = 5
#     repeats = 10
#     cv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats)
#     scorer = make_scorer(balanced_accuracy_score)
#     score = mne.decoding.cross_val_multiscore(
#         PcaLdaClassification(), x_data.__array__(), labels, verbose=10, cv=cv,
#         scoring=scorer, n_jobs=-1)
#     scores.append(np.mean(np.reshape(score, (repeats, kfolds)), axis=1))
#     plt_labels.append(f"{i}\n({reduced.shape[3]})")
# plt.boxplot(scores)
# plt.ylim(0, 1)
# plt.xticks(list(range(1, len(modes)+1)), plt_labels)
# plt.title("Balanced Accuracy")
# plt.xlabel("Trials\n(channels)")
# plt.tight_layout()
# plt.show()
# # %% Confusion matrix
#
# conds = ['aud_ls', 'aud_lm', 'aud_jl']
# idx = sub.AUD
# # conds = ['go_ls', 'go_lm', 'go_jl']
# # idx = sub.PROD
# reduced = sub[:, conds, :, idx]
# reduced.array = reduced.array.dropna()
# # reduced = reduced.nan_common_denom(True, 12, True)
# reduced.smotify_trials()
# comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'trial'))
# # concatenate channels across time as feature vector
# x_data = comb['power'].array.combine((0, 2)).dropna()
# cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
# labels = np.array([cats[k.split('-')[1]] for k in x_data.labels[0]])
# # x_data = np.random.rand(180, 3580)
# matrix = Decoder().cv_cm(x_data, labels, repeats=20)
# fig = ConfusionMatrixDisplay(matrix, display_labels=cats.keys())
# fig.plot(values_format='.3g')


# %% Time Sliding decoding

conds = ['aud_ls', 'aud_lm', 'aud_jl']
# idx = sub.AUD
idxs = [sub.AUD, sub.SM, sub.PROD]
colors = ['g', 'r', 'b']
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None}
fig, ax = plt.subplots()
for i, idx in enumerate(idxs):
    reduced = sub[:, conds][:, :, :, idx]
    reduced.array = reduced.array.dropna()
    reduced = reduced.nan_common_denom(True, 7, False)
    # reduced.smotify_trials()
    comb = reduced.combine(('epoch', 'trial'))['power']
    x_data = (comb.array.dropna()
              # * W[:, 1, None, None]
              ).swapaxes(1, 2)

    # Decoding
    kfolds = 5
    repeats = 5
    mats = Decoder().sliding_window(x_data, 20, -1, kfolds,
                                    repeats, True, None, 6)
    score = mats.T[np.eye(4).astype(bool)].T / np.sum(mats, axis=2)
    scores[list(scores.keys())[i]] = score.copy()
    plot_dist(scores[list(scores.keys())[i]].T, times=(-0.4, 1.4),
              color=colors[i], label=list(scores.keys())[i], ax=ax)
plt.legend()

# # %% Time Generalizing decoding
#
# conds = ['aud_ls', 'aud_lm','aud_jl']
# idx = sub.AUD
# reduced = sub[:, conds, :, idx, :]
# reduced.array = reduced.array.dropna()
# # reduced = reduced.nan_common_denom(True, 12, True)
# # reduced.smotify_trials()
# comb = reduced['power'].combine(('epoch', 'trial'))
# x_data = comb.array.dropna().swapaxes(0, 1)
# cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
# labels = np.array([cats[k.split('-')[0]] for k in x_data.labels[0]])
#
# # Decoding
# scorer = make_scorer(balanced_accuracy_score)
# SDecoder = mne.decoding.GeneralizingEstimator(PcaLdaClassification(), scorer)
# kfolds = 5
# repeats = 10
# cv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats)
# score = mne.decoding.cross_val_multiscore(
#     SDecoder, x_data.__array__(), labels, verbose=10, cv=cv, n_jobs=-1)
# plt.matshow(np.mean(score, axis=0), np.repeat(np.linspace(-0.5, 1.5, 200), 2))
