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
import mne
from tqdm import tqdm


class Decoder(PcaLdaClassification):

    def __init__(self, *args, cv=RepeatedStratifiedKFold, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv = cv

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray, kfolds: int = 5,
              repeats: int = 10, normalize: str = 'true', verbose: bool = True):
        cv = self.cv(n_splits=kfolds, n_repeats=repeats)
        rep = 0
        n_cats = len(set(labels))
        mats = np.zeros((repeats, kfolds, n_cats, n_cats))
        for f, (train_idx, test_idx) in enumerate(cv.split(x_data, labels)):
            fold = f + 1 - rep * kfolds
            print("Fold {} of {}".format(fold, kfolds))
            self.fit(x_data[train_idx], labels[train_idx])
            pred = self.predict(x_data[test_idx])
            mats[rep, fold-1] = confusion_matrix(labels[test_idx], pred)
            if f - rep * kfolds == kfolds - 1:
                rep += 1
        mats = np.mean(np.sum(mats, axis=1), axis=0)
        if normalize == 'true':
            return mats / np.sum(mats, axis=0)
        elif normalize == 'pred':
            return mats / np.sum(mats, axis=1)
        elif normalize == 'all':
            return mats / np.sum(mats)
        else:
            return mats

    def sliding_window(self, x_data: np.ndarray, labels: np.ndarray,
                       window_size: int = 20, axis: int = -1,
                       kfolds: int = 5, repeats: int = 10,
                       normalize: str = 'true') -> np.ndarray:
        x_window = np.lib.stride_tricks.sliding_window_view(
            x_data, window_size, axis, subok=True)
        windows_ax = axis - 1 if axis < 0 else axis
        out = np.zeros((x_window.shape[windows_ax], 4, 4))
        for i in tqdm(range(x_window.shape[windows_ax])):
            in_data = np.take(x_window, i, windows_ax).reshape((x_data.shape[0], -1))
            out[i] = self.cv_cm(in_data, labels, kfolds, repeats, normalize, False)
        return out

    def splits(self, x_data: LabeledArray, folds: int = 5, obs_axs: int = -2,
               smote: bool = True):

        obs_axs = list(range(x_data.ndim))[obs_axs]
        non_trial_dims = tuple(i for i in range(x_data.ndim + 1) if i != obs_axs)

        f_idx = np.random.choice(np.arange(x_data.shape[obs_axs]),
                                 (x_data.shape[obs_axs] // folds, folds),
                                 False)
        f_data = np.take(x_data.__array__(), f_idx, obs_axs)
        while np.sum(np.any(np.isnan(f_data), axis=non_trial_dims) == False) < 1:
            f_idx = np.random.choice(np.arange(x_data.shape[obs_axs]),
                                     (x_data.shape[obs_axs]//folds, folds),
                                     False)
            f_data = np.take(x_data.__array__(), f_idx, obs_axs)

        data = [np.take(x_data, f_idx[..., j], obs_axs) for j in range(folds)]
        for j in range(folds):
            X_test = data[j]
            X_train = np.concatenate(data[:j] + data[j+1:], axis=obs_axs)
            yield X_train, X_test, y_train, y_test



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

conds = ['aud_ls', 'aud_lm','aud_jl']
# idx = sub.SM
idxs = [sub.PROD, sub.AUD, sub.SM]
colors = ['b', 'g', 'r']
scores = {'Production': None, 'Auditory': None, 'Sensory-Motor': None}
for i, idx in enumerate(idxs):
    reduced = sub[:, conds, :, idx, :]
    reduced.array = reduced.array.dropna()
    reduced = reduced.nan_common_denom(True, 10, False)
    # reduced.smotify_trials()
    comb = reduced.combine(('epoch', 'trial'))['power']
    x_data = (comb.array.dropna()
              # * W[:, 1, None, None]
              ).swapaxes(1, 2)
    folds = 5
    obs_axs = 1
    non_trial_dims = tuple(i for i in range(x_data.ndim + 1) if i != obs_axs)

    f_idx = np.random.choice(np.arange(x_data.shape[obs_axs]),
                             (x_data.shape[obs_axs] // folds, folds),
                             False)
    f_data = np.take(x_data, f_idx, obs_axs)
    cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
    # cats = {'go_ls': 1, 'go_lm': 2, 'go_jl': 3}
    labels = np.array([cats[k.split('-')[0]] for k in x_data.labels[0]])

    # Decoding
    kfolds = 5
    repeats = 4
    mats = Decoder().sliding_window(x_data.__array__(), labels, 20, -1, kfolds,
                                    repeats, None)
    score = mats.T[np.eye(4).astype(bool)].T / np.sum(mats, axis=1)
    scores[list(scores.keys())[i]] = score.copy()
    plot_dist(scores[list(scores.keys())[i]].T, times=(-0.4, 1.4),
              color=colors[i], label=list(scores.keys())[i])
plt.legend()

# %% Time Generalizing decoding

conds = ['aud_ls', 'aud_lm','aud_jl']
idx = sub.AUD
reduced = sub[:, conds, :, idx, :]
reduced.array = reduced.array.dropna()
# reduced = reduced.nan_common_denom(True, 12, True)
# reduced.smotify_trials()
comb = reduced['power'].combine(('epoch', 'trial'))
x_data = comb.array.dropna().swapaxes(0, 1)
cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
labels = np.array([cats[k.split('-')[0]] for k in x_data.labels[0]])

# Decoding
scorer = make_scorer(balanced_accuracy_score)
SDecoder = mne.decoding.GeneralizingEstimator(PcaLdaClassification(), scorer)
kfolds = 5
repeats = 10
cv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats)
score = mne.decoding.cross_val_multiscore(
    SDecoder, x_data.__array__(), labels, verbose=10, cv=cv, n_jobs=-1)
plt.matshow(np.mean(score, axis=0), np.repeat(np.linspace(-0.5, 1.5, 200), 2))
