# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification
from ieeg.viz.utils import plot_dist
from ieeg.calc.reshape import smote as do_smote
from joblib import Parallel, delayed


class Decoder(PcaLdaClassification):

    def __init__(self, *args, cv=StratifiedKFold,
                 categories: dict = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cv = cv
        self.categories = categories

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray, kfolds: int = 5,
              repeats: int = 10, smote: bool = True,
              normalize: str = 'true', obs_axs: int = -2):
        cv = self.cv(n_splits=kfolds)
        n_cats = len(set(labels))
        mats = np.zeros((repeats, kfolds, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx_idx = [None if i != obs_axs else slice(None) for i in range(x_data.ndim)]
        for rep in range(repeats):
            for f, (train_idx, test_idx) in enumerate(cv.split(labels, labels)):
                x_train = np.take(x_data, train_idx, obs_axs)
                x_test = np.take(x_data, test_idx, obs_axs)
                y_train = np.take(labels, train_idx, 0)
                y_test = np.take(labels, test_idx, 0)
                if smote:
                    for i in set(labels):
                        idx = np.where(y_train == i)[0][tuple(idx_idx)]
                        x_smote = np.take_along_axis(x_train, idx, obs_axs)
                        np.put_along_axis(x_train, idx, do_smote(x_smote),
                                          obs_axs)
                x_test[np.isnan(x_test)] = (np.random.rand(np.sum(
                    np.isnan(x_test))) - 0.5) * np.nanmean(x_test)

                self.fit(flatten_features(x_train, obs_axs), y_train)
                pred = self.predict(flatten_features(x_test, obs_axs))
                mats[rep, f] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        mats = np.mean(np.sum(mats, axis=1), axis=0)
        if normalize == 'true':
            return mats / np.sum(mats, axis=1).T
        elif normalize == 'pred':
            return mats / np.sum(mats, axis=0)
        elif normalize == 'all':
            return mats / np.sum(mats)
        else:
            return mats

    def sliding_window(self, x_data: np.ndarray, labels: np.ndarray,
                       window_size: int = 20, axis: int = -1, kfolds: int = 5,
                       repeats: int = 10,
                       obs_axs: int = -2, smote: bool = True,
                       normalize: str = 'true', n_jobs: int = -3) -> np.ndarray:

        # make windowing generator
        axis = x_data.ndim + axis if axis < 0 else axis
        slices = (slice(start, start + window_size)
                  for start in range(0, x_data.shape[axis] - window_size))
        idxs = (tuple(slice(None) if i != axis else sl for i in
                      range(x_data.ndim)) for sl in slices)

        # initialize output array
        n_cats = len(self.categories)
        out = np.zeros((x_data.shape[axis] - window_size + 1, n_cats, n_cats))

        # Use joblib to parallelize the computation
        gen = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
            delayed(self.cv_cm)(x_data[idx], labels, kfolds, repeats,
                                smote, normalize, obs_axs) for idx in idxs)
        for i, mat in enumerate(gen):
            out[i] = mat

        return out


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.copy()
    return out.reshape(out.shape[0], -1)


def classes_from_labels(labels: np.ndarray, delim: str = '-', which: int = 0,
                        crop: slice = slice(None)) -> tuple[dict, np.ndarray]:
    class_ids = np.array([k.split(delim)[which][crop] for k in labels])
    classes = {k: i for i, k in enumerate(np.unique(class_ids))}
    return classes, np.array([classes[k] for k in class_ids])

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')


# %% Time Sliding decoding

conds = ['aud_ls', 'aud_lm', 'aud_jl']
# idx = sub.AUD
idxs = [sub.AUD]
colors = ['g', 'r', 'b']
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None}
fig, ax = plt.subplots()
for i, idx in enumerate(idxs):
    reduced = sub[:, conds][:, :, :, idx]
    reduced.array = reduced.array.dropna()
    # also sorts the trials by nan or not
    reduced = reduced.nan_common_denom(True, 6, False)
    comb = reduced.combine(('epoch', 'trial'))['power']
    x_data = (comb.array.dropna()).combine((0, 2))

    cats, labels = classes_from_labels(x_data.labels[1], crop=slice(0, 4))
    _, groups = classes_from_labels(x_data.labels[0], crop=slice(0, 5))

    # Decoding
    kfolds = 5
    repeats = 5
    mats = Decoder().sliding_window(x_data.__array__(), labels, 20, -1, kfolds,
                                    repeats, 1, True, None, 6)
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
