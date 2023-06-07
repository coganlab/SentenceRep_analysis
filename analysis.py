import os
import mne
import numpy as np
from ieeg import PathLike
from ieeg.io import get_data
from ieeg.viz.mri import get_sub_dir
from ieeg.calc.utils import stitch_mats
import matplotlib.pyplot as plt
from utils.mat_load import load_intermediates, group_elecs
from sklearn.decomposition import NMF


class Analysis:
    def __init__(self, d_data: PathLike = None, task: str = "SentenceRep",
                 units: str = "uV", _zscore=None, _significance=None,
                    _power=None, _epochs=None, _names=None):
        mne.set_log_level("ERROR")
        self.root = self.set_root(d_data)
        self.layout = get_data(task, root=self.root)
        self.task = task
        self.units = units
        self.conds = {"resp": (-1, 1),
                      "aud_ls": (-0.5, 1.5),
                      "aud_lm": (-0.5, 1.5),
                      "aud_jl": (-0.5, 1.5),
                      "go_ls": (-0.5, 1.5),
                      "go_lm": (-0.5, 1.5),
                      "go_jl": (-0.5, 1.5)}
        if _power is None or _names is None or _epochs is None:
            epochs, self.power, self.names = load_intermediates(
                self.layout, self.conds, "power")
            self.epochs = {k: v for k, v in epochs.items() if v}
        else:
            self.power = _power
            self.names = _names
            self.epochs = _epochs

        if _significance is None:
            _, self.sig, _ = load_intermediates(
                self.layout, self.conds, "significance")
        else:
            self.sig = _significance

        if _zscore is None:
            _, self.zscore, _ = load_intermediates(
                self.layout, self.conds, "zscore")
        else:
            self.zscore = _zscore

        self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
            self.sig, self.names, self.conds)
        self.subjects = list(self.epochs.keys())
        self.subjects_dir = self.set_subjects_dir()

    @staticmethod
    def set_root(root: PathLike):
        if root is None:
            HOME = os.path.expanduser("~")
            if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
                root = os.path.join(HOME, "workspace", "CoganLab")
            else:  # if not then set box directory
                root = os.path.join(HOME, "Box", "CoganLab")
        return root

    @staticmethod
    def set_subjects_dir(subjects_dir: PathLike = None):
        return get_sub_dir(subjects_dir)

    def __repr__(self):
        return f"Analysis(root={self.root}, task={self.task}, " \
               f"units={self.units})"

    def __getitem__(self, item: str):
        out = self.copy()
        if item in self.subjects:
            out.epochs = self.epochs[item]
            idx = [i for i, n in enumerate(self.names) if item in n]
            out.names = [self.names[i] for i in idx]
            for i, data in enumerate([out.power, out.sig, out.zscore]):
                for k in data.keys():
                    data[k] = data[k][idx, :]
                [out.power, out.sig, out.zscore][i] = data
            for i, data in enumerate([out.AUD, out.SM, out.PROD, out.sig_chans]):
                new = []
                for d in data:
                    if d in idx:
                        new.append(d)
                [out.AUD, out.SM, out.PROD, out.sig_chans][i] = new
        return out

    def copy(self):
        return Analysis(self.root, self.task, self.units,
                        _zscore=self.zscore.copy(),
                        _significance=self.sig.copy(),
                        _power=self.power.copy(),
                        _epochs=self.epochs.copy(),
                        _names=self.names.copy())


if __name__ == "__main__":
    SentenceRep = Analysis()
    x = SentenceRep['D0003']
