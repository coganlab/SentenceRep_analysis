import numpy as np
import os
from bids import BIDSLayout
from ieeg import Doubles, PathLike
import mne
from tqdm import tqdm
from ieeg.calc.mat import LabeledArray
from ieeg.calc.reshape import concatenate_arrays
from collections import OrderedDict


def load_intermediates(layout: BIDSLayout, conds: dict[str, Doubles],
                       value_type: str = "zscore", avg: bool = True,
                       derivatives_folder: PathLike = 'stats') -> (
        dict[dict[str, mne.Epochs]], dict[np.ndarray], list[str]):

    allowed = ["zscore", "power", "significance"]
    match value_type:
        case "zscore":
            reader = lambda f: mne.read_epochs(f, False, preload=True)
            suffix = "zscore-epo"
        case "power":
            reader = lambda f: mne.read_epochs(f, False, preload=True)
            suffix = "power-epo"
        case "significance":
            reader = mne.read_evokeds
            suffix = "mask-ave"
        case _:
            raise ValueError(f"value_type must be one of {allowed}, instead"
                             f" got {value_type}")
    chn_names = []
    epochs = dict()
    all_sig = dict()
    for cond in conds.keys():
        all_sig[cond] = []
    folder = os.path.join(layout.root, 'derivatives', derivatives_folder)
    for subject in tqdm(layout.get_subjects(), desc=f"Loading {value_type}"):
        epochs[subject] = dict()
        for cond in conds.keys():
            try:
                fname = os.path.join(folder, f"{subject}_{cond}_{suffix}.fif")
                epochs[subject][cond] = reader(fname, verbose=False)
            except FileNotFoundError as e:
                mne.utils.logger.warn(e)
                continue
            sig = epochs[subject][cond]
            if suffix.endswith("epo"):
                if avg:
                    sig = sig.average(method=lambda x: np.nanmean(x, axis=0))

            else:
                sig = LabeledArray.from_signal(sig[0])
                sig.prepend_labels(subject + '-', 0)
                sig = sig.to_dict()
            epochs[subject][cond] = sig

        if not epochs[subject]:
            continue
        elif suffix.endswith("epo"):
            epochs[subject] = mne.concatenate_epochs(list(epochs[subject].values()))
            epochs[subject] = LabeledArray.from_signal(epochs[subject]).to_dict()

        epochs = LabeledArray.from_dict(epochs)
        epochs = epochs.combine((0, 2))

    for cond in conds.keys():
        # add new channels to power and significance matrix
        all_sig[cond] = concatenate_arrays(all_sig[cond], -2)

    return epochs, all_sig, chn_names


def load_dict_async(subject: str, suffix: str, reader: callable,
                    conds: dict, folder: PathLike, avg: bool = True):
    out = OrderedDict()
    for cond in conds.keys():
        out[cond] = OrderedDict()
        try:
            fname = os.path.join(folder, f"{subject}_{cond}_{suffix}.fif")
            epoch = reader(fname)
        except FileNotFoundError as e:
            mne.utils.logger.warn(e)
            return

        sig = epoch
        times = conds[cond]
        if suffix.endswith("epo"):
            if avg:
                sig = sig.average(method=lambda x: np.nanmean(x, axis=0))
        else:
            sig = sig[0]
        mat = sig.get_data(tmin=times[0], tmax=times[1])

        # get_data calls are expensive!!!!
        for i, ch in enumerate(sig.ch_names):
            if suffix.endswith("epo"):
                for ev, id in sig.event_id.items():
                    ev = ev.split('/')[-1]
                    out[cond].setdefault(ev, {}).setdefault(ch, {})
                    out[cond][ev][ch] = mat[sig.events[:, 2] == id, i]
            else:
                out[cond][ch] = mat[i]
    return out


def load_dict(layout: BIDSLayout, conds: dict[str, Doubles],
              value_type: str = "zscore", avg: bool = True,
              derivatives_folder: PathLike = 'stats'
              ) -> dict[str: dict[str: dict[str: np.ndarray]]]:

    allowed = ["zscore", "power", "significance"]
    match value_type:
        case "zscore":
            reader = lambda f: mne.read_epochs(f, False, preload=True)
            suffix = "zscore-epo"
        case "power":
            reader = lambda f: mne.read_epochs(f, False, preload=True)
            suffix = "power-epo"
        case "significance":
            reader = mne.read_evokeds
            suffix = "mask-ave"
        case _:
            raise ValueError(f"value_type must be one of {allowed}, instead"
                             f" got {value_type}")
    folder = os.path.join(layout.root, 'derivatives', derivatives_folder)

    subjects = layout.get_subjects()
    subjects.sort()
    subjects = tqdm(subjects, desc=f"Loading {value_type}")
    out = OrderedDict()
    for subject in subjects:
        temp = load_dict_async(subject, suffix, reader, conds, folder, avg)
        if temp is not None:
            out[subject] = temp
    return out


def group_elecs(all_sig: dict[str, np.ndarray] | LabeledArray, names: list[str],
                conds: tuple[str]
                ) -> (list[int], list[int], list[int], list[int]):
    sig_chans = []
    AUD = []
    SM = []
    PROD = []
    for i, name in enumerate(names):
        for cond in conds:
            idx = i
            if np.any(all_sig[cond, idx] == 1):
                sig_chans.append(i)
                break

        if np.squeeze(all_sig).ndim >= 3:
            aud_slice = slice(50, 175)
        else:
            aud_slice = None

        audls_is = np.any(all_sig['aud_ls', idx, aud_slice] == 1)
        audlm_is = np.any(all_sig['aud_lm', idx, aud_slice] == 1)
        audjl_is = np.any(all_sig['aud_jl', idx, aud_slice] == 1)
        mime_is = np.any(all_sig['go_lm', idx] == 1)
        speak_is = np.any(all_sig['go_ls', idx] == 1)

        if audls_is and audlm_is and mime_is and speak_is:
            SM.append(i)
        elif audls_is and audlm_is and audjl_is:
            AUD.append(i)
        elif mime_is and speak_is:
            PROD.append(i)
    return AUD, SM, PROD, sig_chans


def nan_concat(arrs: tuple | list, axis: int = 0) -> np.ndarray:
    """Concatenate arrays, filling in missing values with NaNs"""
    unequal_ax = [ax for ax in range(arrs[0].ndim) if arrs[0].shape[ax] != arrs[1].shape[ax]]
    stretch_ax = [ax for ax in unequal_ax if ax != axis]
    if len(stretch_ax) > 1:
        return np.concatenate(arrs, axis=axis)
    else:
        stretch_ax = stretch_ax[0]
    max_len = max([arr.shape[stretch_ax] for arr in arrs])
    new_arrs = []
    for arr in arrs:
        new_shape = list(arr.shape)
        new_shape[stretch_ax] = max_len
        if 0 in arr.shape:
            continue
        elif arr.shape[stretch_ax] < max_len:
            new_arr = np.full(new_shape, np.nan)

            # fill in the array with the original values
            idx = [slice(None)] * arr.ndim
            idx[stretch_ax] = slice(0, arr.shape[stretch_ax])
            new_arr[*idx] = arr
            new_arrs.append(new_arr)
        else:
            new_arrs.append(arr)
    return np.concatenate(new_arrs, axis=axis)


if __name__ == "__main__":
    from ieeg.io import get_data
    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    conds = {"resp": (-1, 1),
             "aud_ls": (-0.5, 1.5),
             "aud_lm": (-0.5, 1.5),
             "aud_jl": (-0.5, 1.5),
             "go_ls": (-0.5, 1.5),
             "go_lm": (-0.5, 1.5),
             "go_jl": (-0.5, 1.5)}
    epochs, all_power, names = load_intermediates(layout, conds, "power")
    signif, all_sig, _ = load_intermediates(layout, conds, "significance")
