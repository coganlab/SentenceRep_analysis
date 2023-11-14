# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

from analysis.grouping import GroupData
import ieeg.decoding as eegdec
from ieeg.calc.mat import LabeledArray
from ieeg.viz.utils import plot_dist
from ieeg.calc.oversample import oversample_nan, normnd as norm, mixupnd as mixup, TwoSplitNaN
from joblib import Parallel, delayed


class Decoder(eegdec.PcaLdaClassification):

    def __init__(self, *args,
                 cv=TwoSplitNaN,
                 categories: dict = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
                 n_splits: int = 5,
                 n_repeats: int = 10,
                 oversample: bool = True,
                 max_features: int = float("inf"),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cv = cv(n_splits=n_splits, n_repeats=n_repeats)
        self.categories = categories
        self.max_features = max_features

        if not oversample:
            self.oversample = lambda x, func, ax: x
        else:
            self.oversample = lambda x, func, ax: oversample_nan(x, func, ax, False)

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        cv = self.cv
        n_cats = len(set(labels))
        mats = np.zeros((cv.n_repeats, cv.n_splits, n_cats, n_cats))
        auc = np.zeros((cv.n_repeats, cv.n_splits, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(cv.split(x_data.swapaxes(
                0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            for i in set(labels):
                # fill in train data nans with random combinations of
                # existing train data trials (mixup)
                idx[obs_axs] = y_train == i
                x_train[tuple(idx)] = self.oversample(
                    x_train[tuple(idx)], mixup, obs_axs)

                # fill in test data nans with noise from distribution
                # of existing test data
                idx[obs_axs] = y_test == i
                x_test[tuple(idx)] = self.oversample(
                    x_test[tuple(idx)], norm, obs_axs)

            # x_test[np.isnan(x_test)] = np.random.normal(
            #     np.nanmean(x_test), np.nanstd(x_test),
            #     np.sum(np.isnan(x_test)))
            train_in = flatten_features(x_train, obs_axs)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features:
                tidx = np.random.choice(train_in.shape[1], self.max_features,
                                       replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]
            else:
                tidx = slice(None)
            self.fit(train_in, y_train)
            pred = self.predict(test_in)
            rep, fold = divmod(f, cv.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)
            bin_y_test = np.array([y_test == i for i in range(n_cats)]).T
            auc[rep, fold] = roc_auc_score(
                bin_y_test, self.model.decision_function(test_in),
                multi_class='ovr', average=None)

        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1)
        auck = np.mean(auc, axis=1)
        if normalize == 'true':
            divisor = np.sum(matk, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(matk, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = cv.n_repeats #np.sum(matk, keepdims=True)
        else:
            divisor = 1
        return matk / divisor, auck

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
        out = np.zeros((x_data.shape[axis] - window_size, self.cv.n_repeats, n_cats, n_cats))
        out_auc = np.zeros((x_data.shape[axis] - window_size, self.cv.n_repeats, n_cats))

        # Use joblib to parallelize the computation
        gen = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
            delayed(self.cv_cm)(x_data[idx], labels, normalize, obs_axs
                                ) for idx in idxs)
        for i, mat in enumerate(gen):
            out[i], out_auc[i] = mat

        return out, out_auc


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


def extract(sub: GroupData, conds: list[str], idx: list[int] = slice(None), common: int = 5,
            datatype: str = 'zscore', crop_nan: bool = False) -> LabeledArray:
    # if isinstance(idx[0], list):

    reduced = sub[:, conds][:, :, :, idx]
    reduced.array = reduced.array.dropna()
    # also sorts the trials by nan or not
    reduced = reduced.nan_common_denom(True, common, crop_nan)
    comb = reduced.combine(('epoch', 'trial'))[datatype]
    return (comb.array.dropna()).combine((0, 2))

def scale(X, xmax: float, xmin: float):
    return (X - xmin) / (xmax - xmin)

# %% Imports
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')
# sub['power'].array = scale(sub['power'].array, np.max(sub['zscore'].array), np.min(sub['zscore'].array))
all_scores = {}
all_data = []

# %% Time Sliding decoding

conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm']]
# idx = sub.AUD
colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None}
idxs = [sub.AUD, sub.SM, sub.PROD]
# colors = ['c', 'm', 'k', 'orange']
# scores = {'Instructional': None, 'Motor': None, 'Feedback': None, 'Working Memory': None}
# W = np.load('weights.npy')
# groups = [[sub.SM[i] for i in np.where(np.argmax(W, axis=1) == j)[0]]
#               for j in range(W.shape[1])]
# idxs = groups
# idxs = [sub.SM] * 4
idxs = [list(idx & sub.grey_matter) for idx in idxs]
names = list(scores.keys())
fig, axs = plt.subplots(1, len(conds))
fig2, axs2 = plt.subplots(1, len(idxs))
decoder = Decoder(0.8, n_splits=5, n_repeats=5, oversample=True,
                  DA_kwargs={'solver': 'svd', 'store_covariance': True}
                  , max_features=50*30)
scorer = 'acc'
# temp = sub.array['zscore'][..., idxs[0], :, :]
if len(conds) == 1:
    axs = [axs]
    axs2 = [axs2] * len(idxs)
for i, (idx, ax2) in enumerate(zip(idxs, axs2)):
    # sub.array['zscore', :, :, idx] = (temp * W[:, i, None, None]).swapaxes(2, 1).swapaxes(1, 0)
    all_conds = [c for subconds in conds for c in subconds]
    x_data = extract(sub, all_conds, idx, 5, 'zscore', False)
    ax2.set_title(names[i])
    for cond, ax in zip(conds, axs):
        if isinstance(cond, list):
            X = x_data[:, cond[0]]
            for c in cond[1:]:
                X.concatenate(x_data[:, c], axis=1)
            cond = "-".join(cond)
        else:
            X = x_data[:, cond]
        all_data.append(X)

        cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))
        # np.random.shuffle(labels)

        # Decoding
        mats, auc = decoder.sliding_window(X.__array__(), labels, 30, -1, 1,
                                           'true', 6)
        if scorer == 'acc':
            score = mats.T[np.eye(4).astype(bool)].T# [acc_idx] / np.sum(mats, axis=-1)
        else:
            score = auc
        scores[names[i]] = score.copy()
        if cond == 'resp':
            times = (-0.9, 0.9)
        else:
            times = (-0.4, 1.4)
        pl_sc = np.reshape(score.copy(), (score.shape[0], -1)).T
        plot_dist(pl_sc, times=times,
                  color=colors[i], label=list(scores.keys())[i], ax=ax)
        plot_dist(pl_sc, times=times, label=cond, ax=ax2)
        all_scores["-".join([names[i], cond])] = score.copy()

        if i == len(conds) - 1:
            ax.axhline(1/len(set(labels)), color='k', linestyle='--')
            # ax.legend()
            ax.set_title(cond)
            # ax.set_ylim(0.1, 0.8)
    # if i == 0:
    #     ax2.legend()
    # ax2.set_ylim(0.1, 0.8)
    ax2.axhline(1/len(set(labels)), color='k', linestyle='--')

# %% plot the auditory and response aligned decoding

# fig, axs = plt.subplots(1, 2)
# conds = ['aud_ls', 'resp']
# idx = sub.SM
# x_data = extract(sub, conds, idx, 5, 'zscore', False)
# for i, (cond, ax) in enumerate(zip(conds, axs)):
#     X = x_data[:, cond]
#     cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))
#     # np.random.shuffle(labels)
#
#     # Decoding
#     mats, auc = decoder.sliding_window(X.__array__(), labels, 30, -1, 1,
#                                   'true', 7)
#     if scorer == 'acc':
#         score = mats.T[np.eye(4).astype(bool)].T
#     else:
#         score = auc
#     scores[names[i]] = score.copy()
#     if cond == 'resp':
#         times = (-0.9, 0.9)
#     else:
#         times = (-0.4, 1.4)
#     pl_sc = np.reshape(score.copy(), (score.shape[0], -1)).T
#     plot_dist(pl_sc, times=times, color='r', ax=ax)
#     all_scores["-".join([names[i], cond])] = score.copy()
#
#     ax.axhline(1/len(set(labels)), color='k', linestyle='--')
#     ax.set_title(cond)
#     ax.set_ylim(0.1, 0.8)

