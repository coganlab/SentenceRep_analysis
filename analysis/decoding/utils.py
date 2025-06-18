import numpy as np
from typing import Any, Generator

from numpy import ndarray
from sklearn import config_context
from ieeg.decoding.decode import (
    Decoder, flatten_list, classes_from_labels, nan_common_denom
)
from ieeg.arrays.label import LabeledArray
import cupy as cp

def decode_and_score(decoder, data, labels, scorer='acc', **decoder_kwargs):
    """Perform decoding and scoring"""
    mats = decoder.cv_cm(data.__array__(), labels, **decoder_kwargs)
    if scorer == 'acc':
        score = np.mean(mats.T[np.eye(len(decoder.categories)).astype(bool)].T,
                        axis=-1)
    else:
        raise NotImplementedError("Only accuracy is implemented")
    return score

def concatenate_conditions(data, conditions, axis=1, trial_axis=2):
    """Concatenate data for all conditions"""
    concatenated_data = np.take(data, conditions[0], axis=axis)
    for condition in conditions[1:]:
        cond_data = np.take(data, condition, axis=axis)
        concatenated_data = concatenated_data.concatenate(cond_data,
                                                          axis=trial_axis - 1)
    return concatenated_data

def get_scores(array, decoder: Decoder, idxs: list[list[int]],
               conds: list[str], names: list[str], on_gpu: bool = False,
               **decoder_kwargs) -> Generator[ndarray, Any, None]:
    ax = array.ndim - 2
    for i, idx in enumerate(idxs):
        all_conds = flatten_list(conds)
        x_data = extract(array, all_conds, ax, idx, decoder.n_splits,
                         False)

        for cond in conds:
            if isinstance(cond, list):
                X = concatenate_conditions(x_data, cond, 0, ax-1)
                cond = "-".join(cond)
            else:
                X = x_data[cond,].dropna()
            cats, labels = classes_from_labels(X.labels[ax-2],
                                               crop=slice(0, 4),
                                               cats=decoder.categories)

            # Decoding
            decoder.current_job = "-".join([names[i], cond])
            if on_gpu:
                if cp is None:
                    raise ImportError("CuPy is not installed.")
                with config_context(array_api_dispatch=True,
                                    enable_metadata_routing=True,
                                    skip_parameter_validation=True):
                    data = cp.asarray(X.__array__())
                    labels = cp.asarray(labels)
                    score = decoder.cv_cm(data, labels, **decoder_kwargs)
                yield score.get()
            else:
                yield decoder.cv_cm(X.__array__(), labels, **decoder_kwargs)


def extract(array: LabeledArray, conds: list[str], trial_ax: int,
            idx: list[int] = slice(None), common: int = 5,
            crop_nan: bool = False) -> LabeledArray:
    """Extract data from GroupData object"""
    reduced = array[conds, ][:, :, idx]
    reduced = reduced.dropna()
    # also sorts the trials by nan or not
    reduced = nan_common_denom(reduced, True, trial_ax, common, 2, crop_nan)
    comb = reduced.combine((1, trial_ax))
    return comb.dropna()