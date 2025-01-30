import os
from collections import OrderedDict
import mne
import numpy as np
from bids import BIDSLayout
from tqdm import tqdm

from ieeg import Doubles, PathLike
from joblib import Parallel, delayed, cpu_count
from itertools import product

mne.set_log_level("ERROR")

class DataLoader:
    def __init__(self, layout: BIDSLayout, conds: dict[str, Doubles],
                 value_type: str = "zscore", avg: bool = True,
                 derivatives_folder: PathLike = 'stats', ext: str = '.fif'):
        self.root = layout.root
        self.subjects = sorted(layout.get_subjects())
        self.conds = conds
        self.value_type = value_type
        self.avg = avg
        self.derivatives_folder = derivatives_folder
        self.reader, self.suffix = self._get_reader_and_suffix(ext)

    def _get_reader_and_suffix(self, ext):
        allowed = ["zscore", "power", "significance", "pval"]
        assert ext in ('.fif', '.h5'), "ext must be one of ('.fif', '.h5')"

        match self.value_type:
            case "zscore":
                suffix = "zscore"
                if ext == ".fif":
                    suffix += "-epo" + ext
                    reader = lambda f: mne.read_epochs(f, False, preload=True)
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case "power":
                suffix = "power"
                if ext == ".fif":
                    suffix += "-epo" + ext
                    reader = lambda f: mne.read_epochs(f, False, preload=True)
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case "significance":
                suffix = "mask"
                if ext == ".fif":
                    suffix += "-ave" + ext
                    reader = mne.read_evokeds
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case "pval":
                suffix = "pval"
                if ext == ".fif":
                    suffix += "-ave" + ext
                    reader = mne.read_evokeds
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case _:
                raise ValueError(f"value_type must be one of {allowed},"
                                 f" instead got {self.value_type}")
        return reader, suffix

    def load_subject_condition(self, subject, cond, dtype=None):
        out_cond = OrderedDict()
        try:
            fname = os.path.join(self.root, 'derivatives',
                                 self.derivatives_folder,
                                 f"{subject}_{cond}_{self.suffix}")
            epoch = self.reader(fname)
        except (FileNotFoundError, OSError) as e:
            mne.utils.logger.warn(e)
            return subject, cond, None

        sig = epoch
        times = self.conds[cond]
        if (self.suffix.split('.')[0].endswith("epo") or
                isinstance(sig, mne.time_frequency.EpochsTFR)):
            if self.avg:
                sig = sig.average(method=lambda x: np.nanmean(x, axis=0))
        elif isinstance(sig, list):
            sig = sig[0]
        mat = sig.get_data(tmin=times[0], tmax=times[1])
        if dtype is not None:
            mat = mat.astype(dtype)

        for i, ch in enumerate(sig.ch_names):
            if (self.suffix.split('.')[0].endswith("epo") or
                    isinstance(sig, mne.time_frequency.EpochsTFR)):
                for ev, id in sig.event_id.items():
                    ev = ev.split('/')[-1]
                    out_cond.setdefault(ev, {}).setdefault(ch, {})
                    if isinstance(sig, mne.time_frequency.EpochsTFR):
                        for j, f in enumerate(sig.freqs):
                            out_cond[ev][ch].setdefault(f, {})
                            out_cond[ev][ch][f] = mat[sig.events[:, 2] == id, i, j]
                    else:
                        out_cond[ev][ch] = mat[sig.events[:, 2] == id, i]
            elif isinstance(sig, mne.time_frequency.AverageTFR):
                for j, f in enumerate(sig.freqs):
                    out_cond.setdefault(ch, {}).setdefault(f, {})
                    out_cond[ch][f] = mat[i, j]
            else:
                out_cond[ch] = mat[i]
        return subject, cond, out_cond

    def load_dict(self, dtype=None, **kwargs):
        out = OrderedDict()
        combos = product(self.subjects, self.conds.keys())

        # joblib settings with some defaults
        kwargs.setdefault("n_jobs", 1) # cpu_count())
        kwargs.setdefault("return_as", "generator")
        kwargs.setdefault("backend", "loky")
        kwargs.setdefault("verbose", 0)

        proc = Parallel(**kwargs)(delayed(self.load_subject_condition)(
            subject, cond, dtype) for subject, cond in combos)
        for subject, cond, result in tqdm(
                proc,
                total=len(self.subjects) * len(self.conds),
                desc=f"(n_jobs={kwargs['n_jobs']}) Loading {self.value_type}",
                unit="files"):
            if result is not None:
                out.setdefault(subject, OrderedDict())[cond] = result

        return out

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

    data_loader = DataLoader(layout, conds, "power")
    power_data = data_loader.load_dict()

    data_loader = DataLoader(layout, conds, "significance")
    significance_data = data_loader.load_dict()
