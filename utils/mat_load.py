import numpy as np
import os
from bids import BIDSLayout
from ieeg import Doubles, PathLike
import mne


def load_intermediates(layout: BIDSLayout, conds: dict[str, Doubles],
                       value_type: str = "zscore",
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
        all_sig[cond] = np.empty((0, 200))
    folder = os.path.join(layout.root, 'derivatives', derivatives_folder)
    for subject in layout.get_subjects():
        epochs[subject] = dict()
        for cond in conds.keys():
            try:
                epochs[subject][cond] = reader(os.path.join(
                    folder, f"{subject}_{cond}_{suffix}.fif"))
            except FileNotFoundError as e:
                mne.utils.logger.warn(e)
                continue

            if suffix.endswith("epo"):
                sig = epochs[subject][cond].average()
                names = [subject + '-' + ch for ch in sig.ch_names]

                # add new channels to list if not already there
                chn_names = chn_names + [ch for ch in names if
                                         ch not in chn_names]
            else:
                sig = epochs[subject][cond][0]

            # add new channels to power and significance matrix
            all_sig[cond] = np.vstack((all_sig[cond], sig.get_data()))

    return epochs, all_sig, chn_names


def group_elecs(all_sig: dict[str, np.ndarray], names: list[str],
                conds: dict[str, Doubles]
                ) -> (list[int], list[int], list[int], list[int]):
    sig_chans = []
    AUD = []
    SM = []
    PROD = []
    for i, name in enumerate(names):
        for cond in conds.keys():
            if np.any(all_sig[cond][i] == 1):
                sig_chans.append(i)
                break

        audls_is = np.any(all_sig['aud_ls'][i][50:175] == 1)
        audlm_is = np.any(all_sig['aud_lm'][i][50:175] == 1)
        audjl_is = np.any(all_sig['aud_jl'][i][50:175] == 1)
        mime_is = np.any(all_sig['go_lm'][i] == 1)
        speak_is = np.any(all_sig['go_ls'][i] == 1)

        if audls_is and audlm_is and mime_is and speak_is:
            SM.append(i)
        elif audls_is and audlm_is and audjl_is:
            AUD.append(i)
        elif mime_is and speak_is:
            PROD.append(i)
    return AUD, SM, PROD, sig_chans


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
