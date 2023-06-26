import numpy as np
import os
from bids import BIDSLayout
from ieeg import Doubles, PathLike
import mne
from tqdm import tqdm
from ieeg.calc.mat import concatenate_arrays
from collections import OrderedDict
import asyncio
import concurrent.futures


def load_intermediates(layout: BIDSLayout, conds: dict[str, Doubles],
                       value_type: str = "zscore", avg: bool = True,
                       derivatives_folder: PathLike = 'stats') -> (
        dict[dict[str, mne.Epochs]], dict[np.ndarray], list[str]):

    allowed = ["zscore", "power", "significance"]
    match value_type:
        case "zscore":
            reader = mne.read_epochs
            suffix = "zscore-epo"
        case "power":
            reader = mne.read_epochs
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

            avg_func = lambda x: np.nanmean(x, axis=0)
            if suffix.endswith("epo"):
                sig = epochs[subject][cond]
                if avg:
                    sig = sig.average(method=avg_func)

            else:
                sig = epochs[subject][cond][0]
                epochs[subject][cond] = epochs[subject][cond][0]

            names = [subject + '-' + ch for ch in sig.ch_names]

            # add new channels to list if not already there
            chn_names = chn_names + [ch for ch in names if
                                     ch not in chn_names]

            all_sig[cond].append(sig.get_data())

    for cond in conds.keys():
        # add new channels to power and significance matrix
        all_sig[cond] = concatenate_arrays(all_sig[cond], -2)

    return epochs, all_sig, chn_names


async def load_dict_async(subject: str, suffix: str, reader: callable,
                          conds: dict, folder: PathLike, avg: bool = True):
    out = OrderedDict()
    for cond in conds.keys():
        out[cond] = OrderedDict()
        try:
            fname = os.path.join(folder, f"{subject}_{cond}_{suffix}.fif")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                epoch = executor.submit(reader, fname, verbose=False).result()
        except FileNotFoundError as e:
            mne.utils.logger.warn(e)
            continue

        avg_func = lambda x: np.nanmean(x, axis=0)
        if suffix.endswith("epo"):
            sig = epoch
            if avg:
                sig = sig.average(method=avg_func)

        else:
            sig = epoch[0]

        for ch in sig.ch_names:
            if suffix.endswith("epo"):
                out[cond][ch] = dict()
                for ev in sig.event_id.keys():
                    out[cond][ch][ev] = sig.get_data(picks=[ch], item=ev)
            else:
                out[cond][ch] = np.squeeze(sig.get_data(picks=[ch]))
    return out


def load_dict(layout: BIDSLayout, conds: dict[str, Doubles],
              value_type: str = "zscore", avg: bool = True,
              derivatives_folder: PathLike = 'stats'
              ) -> dict[str: dict[str: dict[str: np.ndarray]]]:

    allowed = ["zscore", "power", "significance"]
    match value_type:
        case "zscore":
            reader = mne.read_epochs
            suffix = "zscore-epo"
        case "power":
            reader = mne.read_epochs
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

    async def load_subject_data(subject):
        out[subject] = await load_dict_async(subject, suffix, reader, conds, folder, avg)

        if not any(v for v in out[subject].values()):
            out.pop(subject)

    async def load_data():
        await asyncio.gather(*[load_subject_data(subject) for subject in subjects])

    asyncio.run(load_data())
    return out

def group_elecs(all_sig: dict[str, np.ndarray], names: list[str],
                conds: dict[str, Doubles] | tuple[str]
                ) -> (list[int], list[int], list[int], list[int]):
    sig_chans = []
    AUD = []
    SM = []
    PROD = []
    if isinstance(conds, tuple):
        conds = {c: None for c in conds}
    for i, name in enumerate(names):
        for cond in conds.keys():
            if isinstance(all_sig[cond], dict):
                idx = name
            else:
                idx = i
            if np.any(all_sig[cond][idx] == 1):
                sig_chans.append(i)
                break

        audls_is = np.any(all_sig['aud_ls'][idx][50:175] == 1)
        audlm_is = np.any(all_sig['aud_lm'][idx][50:175] == 1)
        audjl_is = np.any(all_sig['aud_jl'][idx][50:175] == 1)
        mime_is = np.any(all_sig['go_lm'][idx] == 1)
        speak_is = np.any(all_sig['go_ls'][idx] == 1)

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
