import numpy as np
from typing import Any, Generator

from numpy import ndarray
from sklearn import config_context
from ieeg.decoding.decode import (
    Decoder, flatten_list, classes_from_labels, nan_common_denom
)
from ieeg.arrays.label import LabeledArray, normalize_index
try:
    import torch
except ImportError:
    torch = None

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
               crop: slice = slice(None), which: int = 0, **decoder_kwargs) -> Generator[ndarray, Any, None]:
    ax = array.ndim - 2
    for i, idx in enumerate(idxs):
        all_conds = flatten_list(conds)
        x_data = extract(array, all_conds, ax, idx, min(3, decoder.n_splits),
                         False)

        for cond in conds:
            if isinstance(cond, list):
                X = x_data[cond].combine((0, ax-1))
                cond = "-".join(cond)
            else:
                X = x_data[cond,].dropna()
            cats, labels = classes_from_labels(X.labels[ax-2],
                                               crop=crop, which=which)
                                               # cats=decoder.categories)
            decoder.categories = cats
            print(f"Decoding {names[i]} {cond} with {len(cats)} classes and ")
            print(f"Categories: {cats}")

            # Decoding
            decoder.current_job = "-".join([names[i], cond])

            if on_gpu:
                if torch is None:
                    raise ImportError("CuPy is not installed.")
                with config_context(array_api_dispatch=True,
                                    enable_metadata_routing=True,
                                    skip_parameter_validation=True):
                    data = torch.from_numpy(X.__array__()).to('cuda')
                    labels = torch.from_numpy(labels).to('cuda')
                    score = decoder.cv_cm(data, labels, **decoder_kwargs)
                yield score.detach().cpu().numpy()
            else:
                yield decoder.cv_cm(np.array(X), labels, **decoder_kwargs)


def extract(array: LabeledArray, conds: list[str], trial_ax: int,
            idx: list[int] = slice(None), common: int = 5,
            crop_nan: bool = False) -> LabeledArray:
    """Extract data from GroupData object"""
    cond_coords = normalize_index(([array.find(cond, 0)
                                    for cond in conds],))
    chan_coords = normalize_index((slice(None), slice(None), idx))
    print(f"Extracting {len(idx)} channels and {len(conds)} conditions")
    reduced = array[cond_coords][chan_coords].dropna()
    # also sorts the trials by nan or not
    print(f"Data shape before nan reduction: {reduced.shape}")
    reduced = nan_common_denom(reduced, True, trial_ax, common, 2, crop_nan)
    print(f"Data shape after nan reduction: {reduced.shape}")
    # combine conditions back into one axis
    comb = reduced.combine((1, trial_ax))
    return comb.dropna()