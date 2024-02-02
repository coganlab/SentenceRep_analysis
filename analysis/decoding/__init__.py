from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from analysis.grouping import GroupData
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from ieeg.calc.oversample import MinimumNaNSplit
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt
from ieeg.viz.utils import plot_dist
from joblib import Parallel, delayed
import itertools


class Decoder(PcaLdaClassification, MinimumNaNSplit):

    def __init__(self, categories: dict, *args,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 oversample: bool = True,
                 max_features: int = float("inf"),
                 min_samples: int = 1,
                 which: str = 'test',
                 **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats,
                                 None, min_samples, which)
        if not oversample:
            self.oversample = lambda x, axis: x
        self.categories = categories
        self.max_features = max_features
        self.obs_axs = None

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2, n_jobs: int = 1,
              average_repetitions: bool = True, window: int = None,
              shuffle: bool = False):
        """Cross-validated confusion matrix"""
        n_cats = len(set(labels))
        out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)
        if window is not None:
            out_shape = (x_data.shape[-1] - window + 1,) + out_shape
        mats = np.zeros(out_shape)
        self.obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs

        if shuffle:
            # shuffled label pool
            label_stack = [labels.copy() for _ in range(self.n_repeats)]
            for i in range(self.n_repeats):
                self.shuffle_labels(x_data, label_stack[i])

            # build the test/train indices from the shuffled labels for each
            # repetition, then chain together the repetitions
            # splits = (train, test)
            idxs = ((self.split(x_data.swapaxes(0, obs_axs), label), label) for
                    label in label_stack)
            idxs = ((splits, itertools.repeat(label, self.n_splits)) for splits, label in idxs)
            splits, label = zip(*idxs)
            splits = itertools.chain.from_iterable(splits)
            label = itertools.chain.from_iterable(label)
            idxs = zip(splits, label)

        else:
            idxs = ((splits, labels) for splits in self.split(x_data.swapaxes(0, obs_axs), labels))

        # loop over folds and repetitions
        results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
            delayed(self.process_fold)(train_idx, test_idx, x_data, l, window)
            for (train_idx, test_idx), l in idxs)

        # Collect the results
        for i, result in enumerate(results):
            rep, fold = divmod(i, self.n_splits)
            mats[:, rep, fold] = result

        # average the repetitions
        if average_repetitions:
            mats = np.mean(mats, axis=1)

        # normalize, sum the folds
        mats = np.sum(mats, axis=-3)
        if normalize == 'true':
            divisor = np.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return mats / divisor

    def process_fold(self, train_idx: np.ndarray, test_idx: np.ndarray,
                     x_data: np.ndarray, labels: np.ndarray,
                     window: int | None):

        # make first and only copy of x_data
        idx_stacked = np.concatenate((train_idx, test_idx))
        x_stacked = np.take(x_data, idx_stacked, self.obs_axs)
        y_stacked = labels[idx_stacked]

        # define train and test as views of x_stacked
        sep = train_idx.shape[0]
        x_train, x_test = np.split(x_stacked, [sep], axis=self.obs_axs)
        y_train, y_test = np.split(y_stacked, [sep])

        idx = [slice(None) for _ in range(x_data.ndim)]
        for i in np.unique(labels):
            # fill in train data nans with random combinations of
            # existing train data trials (mixup)
            idx[self.obs_axs] = y_train == i
            x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)],
                                                  axis=self.obs_axs)

        # fill in test data nans with noise from distribution
        is_nan = np.isnan(x_test)
        x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

        windowed = windower(x_stacked, window, axis=-1)
        out = np.zeros((windowed.shape[0], len(self.categories),
                        len(self.categories)), dtype=np.uintp)
        for i, x_window in enumerate(windowed):
            win_train, win_test = np.split(x_window, [sep], axis=self.obs_axs)
            out[i] = self.fp(win_train, win_test, y_train, y_test)
        return out

    def fp(self, x_train, x_test, y_train, y_test):

        # feature selection
        train_in = flatten_features(x_train, self.obs_axs)
        test_in = flatten_features(x_test, self.obs_axs)
        if train_in.shape[1] > self.max_features:
            tidx = np.random.choice(train_in.shape[1], self.max_features,
                                    replace=False)
            train_in = train_in[:, tidx]
            test_in = test_in[:, tidx]

        # fit model and score results
        self.fit(train_in, y_train)
        pred = self.predict(test_in)
        return confusion_matrix(y_test, pred)


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.view()
    return out.reshape(out.shape[0], -1)


def windower(x_data: np.ndarray, window_size: int, axis: int = -1, insert_at: int = 0):
    """Create a sliding window view of the array with the given window size."""
    # Compute the shape and strides for the sliding window view
    shape = list(x_data.shape)
    shape[axis] = x_data.shape[axis] - window_size + 1
    shape.insert(axis, window_size)
    strides = list(x_data.strides)
    strides.insert(axis, x_data.strides[axis])

    # Create the sliding window view
    out = as_strided(x_data, shape=shape, strides=strides)

    # Move the window size dimension to the front
    out = np.moveaxis(out, axis, insert_at)

    return out


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
    mats = decoder.cv_cm(data.__array__(), labels, **decoder_kwargs)
    if scorer == 'acc':
        score = mats.T[np.eye(len(decoder.categories)).astype(bool)].T
    else:
        raise NotImplementedError("Only accuracy is implemented")
    return score


def get_scores(subjects, decoder, idxs: list[list[int]], conds: list[str],
               **decoder_kwargs) -> dict[str, np.ndarray]:
    all_scores = {}
    names = ['Auditory', 'Sensory-Motor', 'Production', 'All']
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
            score = decoder.cv_cm(X.__array__(), labels, **decoder_kwargs)
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
