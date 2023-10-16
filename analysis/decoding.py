# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _RepeatedSplits
import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification
from ieeg.viz.utils import plot_dist
from ieeg.calc.reshape import oversample_nan, norm, mixup
from joblib import Parallel, delayed


class AtLeastNDataPoints(StratifiedKFold):
    """A StratifiedKFold iterator that requires at least N non-nan data points
    in each stratification of each fold"""

    def __init__(self, n_splits, minimum_trials=1,
                 random_state=None, shuffle=True):
        super().__init__(n_splits=n_splits,
                         random_state=random_state,
                         shuffle=shuffle)
        self.minimum_trials = minimum_trials

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        i = 0
        while self._check_test_folds(X, y, test_folds) is False:
            test_folds = self._make_test_folds(X, y)
            i += 1
            if i > 10000:
                raise RecursionError('Could not find a valid split')
        for i in range(self.n_splits):
            yield test_folds == i

    def _check_test_folds(self, X, y, test_folds):
        for fold in set(test_folds):
            for i in set(y):
                idx = np.logical_and(test_folds == fold, y == i)
                if (np.sum(~np.isnan(X[idx]).any(
                        axis=tuple(range(X.ndim))[1:])) < self.minimum_trials):
                    return False
        return True


class Repeater(_RepeatedSplits):
    """Implements a class that repeats a cross-validation strategy n times"""

    def __init__(self, cv, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(cv,
                         n_repeats=n_repeats,
                         random_state=random_state,
                         n_splits=n_splits)


class Decoder(PcaLdaClassification):

    def __init__(self, *args,
                 cv=AtLeastNDataPoints,
                 categories: dict = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
                 n_splits: int = 5,
                 n_repeats: int = 10,
                 oversample: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cv = Repeater(cv, n_splits=n_splits, n_repeats=n_repeats)
        self.categories = categories

        if not oversample:
            self.oversample = lambda x: x
        else:
            self.oversample = lambda x, func: oversample_nan(x, func, False)

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = 'true', obs_axs: int = -2):
        cv = self.cv
        n_cats = len(set(labels))
        mats = np.zeros((repeats, kfolds, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(cv.split(x_data.swapaxes(
                0, obs_axs), labels)):
            rep = f // kfolds
            fold = f % kfolds
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            y_train = np.take(labels, train_idx, 0)
            y_test = np.take(labels, test_idx, 0)
            for i in set(labels):
                # fill in train data nans with random combinations of
                # existing train data trials (mixup)
                idx[obs_axs] = y_train == i
                x_train[tuple(idx)] = self.oversample(
                    x_train[tuple(idx)], mixup)

                # fill in test data nans with noise from distribution
                # of existing test data
                idx[obs_axs] = y_test == i
                x_test[tuple(idx)] = self.oversample(
                    x_test[tuple(idx)], norm)

            # x_test[np.isnan(x_test)] = np.random.normal(
            #     np.nanmean(x_test), np.nanstd(x_test),
            #     np.sum(np.isnan(x_test)))

            self.fit(flatten_features(x_train, obs_axs), y_train)
            pred = self.predict(flatten_features(x_test, obs_axs))
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        mats = np.sum(mats, axis=1)
        if normalize == 'true':
            return (mats.swapaxes(-2, -1) / np.sum(mats, axis=2, keepdims=True
                                                   )).swapaxes(-2, -1)
        elif normalize == 'pred':
            return mats / np.sum(mats, axis=1, keepdims=True)
        elif normalize == 'all':
            return mats / np.sum(mats)
        else:
            return mats

    def sliding_window(self, x_data: np.ndarray, labels: np.ndarray,
                       window_size: int = 20, axis: int = -1,
                       obs_axs: int = -2, normalize: str = 'true',
                       n_jobs: int = -3) -> np.ndarray:

        # make windowing generator
        axis = x_data.ndim + axis if axis < 0 else axis
        slices = (slice(start, start + window_size)
                  for start in range(0, x_data.shape[axis] - window_size))
        idxs = (tuple(slice(None) if i != axis else sl for i in
                      range(x_data.ndim)) for sl in slices)

        # initialize output array
        n_cats = len(self.categories)
        out = np.zeros((x_data.shape[axis] - window_size, repeats, n_cats, n_cats))

        # Use joblib to parallelize the computation
        gen = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
            delayed(self.cv_cm)(x_data[idx], labels, normalize, obs_axs
                                ) for idx in idxs)
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
    class_ids = np.array([k.split(delim, )[which][crop] for k in labels])
    classes = {k: i for i, k in enumerate(np.unique(class_ids))}
    return classes, np.array([classes[k] for k in class_ids])

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')


# %% Time Sliding decoding

conds = ['aud_lm']
# idx = sub.AUD
idxs = [sub.AUD, sub.SM, sub.PROD]
colors = ['g', 'r', 'b']
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None}
fig, ax = plt.subplots()
for i, idx in enumerate(idxs):
    reduced = sub[:, conds][:, :, :, idx]
    reduced.array = reduced.array.dropna()
    # also sorts the trials by nan or not
    reduced = reduced.nan_common_denom(True, 5, False)
    comb = reduced.combine(('epoch', 'trial'))['zscore']
    x_data = (comb.array.dropna()).combine((0, 2))

    cats, labels = classes_from_labels(x_data.labels[1], crop=slice(0, 4))
    _, groups = classes_from_labels(x_data.labels[0], crop=slice(0, 5))

    # np.random.shuffle(labels)

    # Decoding
    kfolds = 5
    repeats = 4
    decoder = Decoder(n_splits=kfolds, n_repeats=repeats, oversample=True)
    mats = decoder.sliding_window(x_data.__array__(), labels, 20, -1, 1,
                                  'true', 6)
    score = mats.T[np.eye(4).astype(bool)].T
    scores[list(scores.keys())[i]] = score.copy()
    if all(c == 'resp' for c in conds):
        times = (-0.9, 0.9)
    else:
        times = (-0.4, 1.4)
    pl_sc = scores[list(scores.keys())[i]]
    plot_dist(np.reshape(pl_sc, (pl_sc.shape[0], -1)).T, times=times,
              color=colors[i], label=list(scores.keys())[i], ax=ax)
plt.axhline(1/len(set(labels)), color='k', linestyle='--')
plt.legend()
plt.title("Listen-Mime")
plt.ylim(0.1, 0.8)
