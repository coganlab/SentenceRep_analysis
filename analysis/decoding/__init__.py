from ieeg.decoding.decode import flatten_list, concatenate_conditions, classes_from_labels
from ieeg.arrays.label import LabeledArray
import numpy as np

def get_scores(array, decoder, idxs: list[list[int]], conds: list[str],
               names: list[str], weights: list[list[int]] = None, **decoder_kwargs) -> dict[str, np.ndarray]:
    for i, idx in enumerate(idxs):
        all_conds = flatten_list(conds)
        x_data = extract(array, all_conds, idx, decoder.n_splits,
                         False)

        for cond in conds:
            if isinstance(cond, list):
                X = concatenate_conditions(x_data, cond)
                cond = "-".join(cond)
            else:
                X = x_data[:, cond]

            cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))

            # Decoding
            if weights is None:
                score = decoder.cv_cm(X.__array__(), labels, **decoder_kwargs)
                yield "-".join([names[i], cond]), score
            else:
                for j, weight in enumerate(weights):
                    score = decoder.cv_cm(X.__array__() * weight[X.labels[0], None, None].__array__(), labels, **decoder_kwargs)
                    yield "-".join([names[i], cond, str(j)]), score


def extract(array, conds: list[str], idx: list[int] = slice(None), common: int = 5,
            crop_nan: bool = False) -> LabeledArray:
    """Extract data from GroupData object"""
    # reduced = sub[:, conds][:, :, :, idx]
    reduced = array[conds,][:,:,idx]
    reduced = reduced.dropna()
    # also sorts the trials by nan or not
    reduced = nan_common_denom(reduced, True, 3, common, 2, crop_nan)
    comb = reduced.combine((0, 2))
    return (comb.array.dropna()).combine((0, 2))


def nan_common_denom(array: LabeledArray, sort: bool = True, trials_ax: int = 1, min_trials: int = 0,
                     ch_ax: int = 0, crop_trials: bool = True, verbose: bool = False):
    """Remove trials with NaNs from all channels"""
    others = [i for i in range(array.ndim) if ch_ax != i != trials_ax]
    isn = np.isnan(array.__array__())
    nan_trials = np.any(isn, axis=tuple(others))

    # Sort the trials by whether they are nan or not
    if sort:
        order = np.argsort(nan_trials, axis=1)
        old_shape = list(order.shape)
        new_shape = [1 if ch_ax != i != trials_ax else old_shape.pop(0)
                     for i in range(array.ndim)]
        order = np.reshape(order, new_shape)
        data = np.take_along_axis(array.__array__(), order, axis=trials_ax)
        data = LabeledArray(data, array.labels.copy())
    else:
        data = array

    ch_tnum = array.shape[trials_ax] - np.sum(nan_trials, axis=1)
    ch_min = ch_tnum.min()
    if verbose:
        print(f"Lowest trials {ch_min} at "
              f"{array.keys['channel'][ch_tnum.argmin()]}")

    ntrials = max(ch_min, min_trials)
    if ch_min < min_trials:
        # data = data.take(np.where(ch_tnum >= ntrials)[0], ch_idx)
        ch = np.array(array.keys['channel'])[ch_tnum < ntrials].tolist()
        if verbose:
            print(f"Channels excluded (too few trials): {ch}")

    # data = data.take(np.arange(ntrials), trials_idx)
    idx = [np.arange(ntrials) if i == trials_ax and crop_trials
           else np.arange(s) for i, s in enumerate(array.shape)]
    idx[ch_ax] = np.where([ch_tnum >= ntrials])[1]

    return data[np.ix_(*idx)]

