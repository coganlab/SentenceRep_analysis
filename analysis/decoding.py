# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from ieeg.viz.utils import plot_dist
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.process import sliding_window


class Decoder(PcaLdaClassification, MinimumNaNSplit):

    def __init__(self, categories: dict, *args,
                 n_splits: int = 5,
                 n_repeats: int = 10,
                 oversample: bool = True,
                 max_features: int = float("inf"),
                 **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats)
        if not oversample:
            self.oversample = lambda x, axis: x
        self.categories = categories
        self.max_features = max_features

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(
                0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            for i in set(labels):
                # fill in train data nans with random combinations of
                # existing train data trials (mixup)
                idx[obs_axs] = y_train == i
                x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)],
                                                      axis=obs_axs)

                # fill in test data nans with noise from distribution
                # of existing test data
                # idx[obs_axs] = y_test == i
                # x_test[tuple(idx)] = self.oversample(
                #     x_test[tuple(idx)], norm, obs_axs)

            # fill in test data nans with noise from distribution
            # TODO: extract distribution from channel baseline
            is_nan = np.isnan(x_test)
            x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

            # feature selection
            train_in = flatten_features(x_train, obs_axs)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features:
                tidx = np.random.choice(train_in.shape[1], self.max_features,
                                        replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]

            # fit model and score results
            self.fit(train_in, y_train)
            pred = self.predict(test_in)
            rep, fold = divmod(f, self.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1)
        if normalize == 'true':
            divisor = np.sum(matk, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(matk, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return matk / divisor


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


def flatten_list(nested_list: list[list[str] | str]) -> list[str]:
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    return flat_list


def concatenate_conditions(data, conditions, axis=1):
    """Concatenate data for all conditions"""
    concatenated_data = data[:, conditions[0]]
    for condition in conditions[1:]:
        concatenated_data = concatenated_data.concatenate(data[:, condition], axis=axis)
    return concatenated_data


def decode_and_score(decoder, data, labels, scorer='acc', **decoder_kwargs):
    """Perform decoding and scoring"""
    mats = sliding_window(data.__array__(), labels, decoder.cv_cm, **decoder_kwargs)
    if scorer == 'acc':
        score = mats.T[np.eye(len(decoder.categories)).astype(bool)].T
    else:
        raise NotImplementedError("Only accuracy is implemented")
    return score


# %% Imports
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')
# sub['power'].array = scale(sub['power'].array, np.max(sub['zscore'].array), np.min(sub['zscore'].array))
all_scores = {}
all_data = []
colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
idxs = [list(idx & sub.grey_matter) for idx in idxs]
names = list(scores.keys())
decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', n_splits=5, n_repeats=5, oversample=True)
scorer = 'acc'
window_kwargs = {'window_size': 20, 'axis': -1, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2}

# %% Time Sliding decoding for word tokens
shuffle = True
conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
fig, axs = plt.subplots(1, len(conds))
fig2, axs2 = plt.subplots(1, len(idxs))
if len(conds) == 1:
    axs = [axs]
    axs2 = [axs2] * len(idxs)
for i, (idx, ax2) in enumerate(zip(idxs, axs2)):
    all_conds = flatten_list(conds)
    x_data = extract(sub, all_conds, idx, decoder.n_splits, 'zscore', False)
    ax2.set_title(names[i])
    for cond, ax in zip(conds, axs):
        if isinstance(cond, list):
            X = concatenate_conditions(x_data, cond)
            cond = "-".join(cond)
        else:
            X = x_data[:, cond]
        all_data.append(X)

        cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))
        # check that each label has at least 3 non nan trials (So that there is always a way to
        # get 2 trials from 4/5ths of the labels. if not, reshuffle
        if shuffle:
            gt_labels = [0] * len(cats)
            while not all(g >= decoder.n_splits for g in gt_labels):
                np.random.shuffle(labels)
                for j, l in enumerate(cats.values()):
                    gt_labels[j] = np.min(np.sum(np.all(~np.isnan(X[:, labels==l]), axis=2), axis=1))
            del gt_labels

        # Decoding
        score = decode_and_score(decoder, X, labels, scorer, **window_kwargs)
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
            ax.legend()
            ax.set_title(cond)
            ax.set_ylim(0.1, 0.8)
    if i == 0:
        ax2.legend()
    ax2.set_ylim(0.1, 0.8)
    ax2.axhline(1/len(set(labels)), color='k', linestyle='--')


# # %% Time Sliding decoding for conditions
# # conds_aud = ['aud_ls', 'aud_lm', 'aud_jl']
# # conds_go = ['go_ls', 'go_lm', 'go_jl']
# conds_aud = ['aud_ls', 'aud_lm']
# conds_go = ['go_ls', 'go_lm']
# fig2, ax = plt.subplots(1, 2)
# for i, idx in enumerate(idxs):
#
#     # organize data
#     all_data = extract(sub, conds_aud + conds_go, idx, 5, 'zscore', False)
#     aud_data = concatenate_conditions(all_data, conds_aud, 1)
#     aud_data.labels[1] = Labels([l.replace('aud_', '') for l in aud_data.labels[1]])
#     go_data = concatenate_conditions(all_data, conds_go)
#     go_data.labels[1] = Labels([l.replace('go_', '') for l in go_data.labels[1]])
#     common = np.array([l for l in aud_data.labels[1] if l in go_data.labels[1]])
#     x_data = aud_data[..., :175].concatenate(go_data[:, common], axis=2)
#     cats, labels = classes_from_labels(x_data.labels[1], crop=slice(-2, None), which=1)
#     # cats, labels = classes_from_labels(x_data.labels[1], crop=slice(-2, -1), which=1)
#     # cats['ls'] = cats['lm'] # if you want to combine ls and lm
#     decoder.categories = cats
#
#     # Decoding
#     score = decode_and_score(decoder, x_data, labels, scorer, **window_kwargs)
#     scores[names[i]] = score.copy()
#     pl_sc = np.reshape(scores[names[i]], (scores[names[i]].shape[0], -1)).T
#
#     plot_dist(pl_sc[:, :165], times=(-0.4, 1.25), color=colors[i], label=list(scores.keys())[i], ax=ax[0])
#     plot_dist(pl_sc[:, 166:], times=(-0.5, 1.4), color=colors[i],
#               label=list(scores.keys())[i], ax=ax[1])
ax[0].set_xlabel("Time from stim (s)")
ax[1].set_xlabel("Time from go (s)")
ax[0].set_ylabel("Accuracy (%)")
fig.suptitle("Word Decoding")

# # draw horizontal dotted lines at chance
# ax[0].axhline(1/2, color='k', linestyle='--')
# ax[1].axhline(1/2, color='k', linestyle='--')