from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from analysis.grouping import GroupData
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.process import sliding_window
import numpy as np
import matplotlib.pyplot as plt
from ieeg.viz.utils import plot_dist


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
              normalize: str = None, obs_axs: int = -2,
              average_repetitions: bool = True, shuffle: bool = False):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(
                0, obs_axs), labels)):
            rep, fold = divmod(f, self.n_splits)

            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            if shuffle:
                self.shuffle_labels(x_train, y_train, min=2)
                np.random.shuffle(y_test)
                # self.shuffle_labels(x_test, y_test, min=1)

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
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions
        if average_repetitions:
            mats = np.mean(mats, axis=0)
        mats = np.sum(mats, axis=-3)

        # normalize, sum the folds
        if normalize == 'true':
            divisor = np.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return mats / divisor


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
    """Extract data from GroupData object"""
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


def get_scores(subjects, decoder, idxs: list[list[int]], conds: list[str],
               **decoder_kwargs) -> dict[str, np.ndarray]:
    all_scores = {}
    scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None,
              'All': None}
    names = list(scores.keys())
    for i, idx in enumerate(idxs):
        all_conds = flatten_list(conds)
        x_data = extract(subjects, all_conds, idx, decoder.n_splits, 'zscore',
                         False)
        for cond in conds:
            if isinstance(cond, list):
                X = concatenate_conditions(x_data, cond)
                cond = "-".join(cond)
            else:
                X = x_data[:, cond]

            cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))

            # Decoding
            score = sliding_window(X.__array__(), labels, decoder.cv_cm,
                                   **decoder_kwargs)
            scores[names[i]] = score.copy()
            all_scores["-".join([names[i], cond])] = score.copy()
    return all_scores


def plot_all_scores(all_scores: dict[str, np.ndarray],
                    conds: list[str], idxs: dict[str, list[int]],
                    colors: list[list[float]],
                    fig: plt.Figure = None, axs: plt.Axes = None,
                    **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    names = list(idxs.keys())
    if fig is None:
        fig, axs = plt.subplots(1, len(conds))
    if len(conds) == 1:
        axs = [axs]
    for color, name, idx in zip(colors, names, idxs.values()):
        for cond, ax in zip(conds, axs):
            if isinstance(cond, list):
                cond = "-".join(cond)
            ax.set_title(cond)
            if cond == 'resp':
                times = (-0.9, 0.9)
            else:
                times = (-0.4, 1.4)
            pl_sc = np.reshape(all_scores["-".join([name, cond])],
                               (all_scores["-".join([name, cond])].shape[0],
                                -1)).T
            plot_dist(pl_sc, mode='std', times=times,
                      color=color, label=name, ax=ax,
                      **plot_kwargs)
            if name is names[-1]:
                ax.legend()
                ax.set_title(cond)
                ax.set_ylim(0.1, 0.7)
    axs[0].set_xlabel("Time from stim (s)")
    axs[1].set_xlabel("Time from go (s)")
    axs[2].set_xlabel("Time from response (s)")
    axs[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Word Decoding")
    return fig, axs
