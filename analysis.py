import os
import mne
import numpy as np
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from ieeg.viz.utils import plot_weight_dist
from ieeg.viz.mri import get_sub_dir, plot_on_average
from plotting import compare_subjects, plot_clustering
from utils.mat_load import load_intermediates, group_elecs
from copy import deepcopy
from sklearn.decomposition import NMF


class GroupData:
    def __init__(self, d_data: PathLike = None, task: str = "SentenceRep",
                 units: str = "uV", conditions: dict[str, Doubles] = None,
                 _zscore=None, _significance=None, _power=None, _epochs=None,
                 _names=None):
        mne.set_log_level("ERROR")
        self._layout = get_data(task, root=self._set_root(d_data))
        self.task = task
        self.units = units
        self.conds = self._set_conditions(conditions)

        if _significance is None or _names is None or _epochs is None:
            epochs, self.sig, self.names = load_intermediates(
                self._layout, self.conds, "significance")
            self.epochs = {k: v for k, v in epochs.items() if v}
        else:
            self.sig = _significance
            self.names = _names
            self.epochs = _epochs

        self.power = self._load_intermediates(_power, "power")
        self.zscore = self._load_intermediates(_zscore, "zscore")

        if all(cond in self.conds.keys()
               for cond in ["aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm"]):
            self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
                self.sig, self.names, self.conds)
        else:
            self.sig_chans = self._find_sig_chans(self.sig)

        self.subjects = list(set(n[:5] for n in self.names))
        self.subjects.sort()
        self.subjects_dir = self._set_subjects_dir()
        self.shape = self._get_shape()
        self.size = np.prod(self.shape)

    @staticmethod
    def _set_root(root: PathLike):
        if root is None:
            HOME = os.path.expanduser("~")
            if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
                root = os.path.join(HOME, "workspace", "CoganLab")
            else:  # if not then set box directory
                root = os.path.join(HOME, "Box", "CoganLab")
        return root

    @staticmethod
    def _set_conditions(conditions: dict[str, Doubles]):
        if conditions is None:
           return {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                   "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                   "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                   "go_jl": (-0.5, 1.5)}
        else:
            return conditions

    @staticmethod
    def _set_subjects_dir(subjects_dir: PathLike = None):
        return get_sub_dir(subjects_dir)

    @staticmethod
    def _find_sig_chans(sig: np.ndarray) -> list[int]:
        return np.where(np.any(sig == 1, axis=1))[0].tolist()

    def _load_intermediates(self, input_data: dict, attr: str):
        if input_data is None:
            _, val, _ = load_intermediates(
                self.layout, self.conds, attr)
        else:
            val = input_data
        return val

    def _get_shape(self):
        if isinstance(self.sig, dict):
            return list(self.sig.values())[0].shape
        elif isinstance(self.sig, np.ndarray):
            return self.sig.shape
        else:
            raise ValueError(f"Invalid type for sig {type(self.sig)}")

    def __repr__(self):
        return f"Analysis(root={self.root}, task={self.task}, " \
               f"units={self.units})"

    def __getitem__(self, item: str):

        if item in self.subjects:
            out = self.get_subject(item)
        elif item in list(self.conds.keys()):
            out = self.get_condition(item)
        return out

    def copy(self):
        del self.layout
        copy = deepcopy(self)
        self.layout = copy.layout = get_data(self.task, root=self.root)
        return copy

    def get_subject(self, sub_id: str):
        assert sub_id in self.subjects
        epochs = self.epochs[sub_id]
        idx = [i for i, n in enumerate(self.names) if sub_id in n]
        names = [self.names[i] for i in idx]

        new_data = {}
        for at in ["power", "sig", "zscore"]:
            data = getattr(self, at)
            new_data[at] = dict()
            for k in data.keys():
                new_data[at][k] = data[k][idx, :]
        power = new_data["power"]
        sig = new_data["sig"]
        zscore = new_data["zscore"]

        return type(self)(self.root, self.task, self.units, self.conds, zscore,
                        sig, power, epochs, names)

    def get_condition(self, condition: str):
        assert condition in self.conds.keys()
        if condition in self.epochs.keys():
            epochs = self.epochs[condition]
        elif any(sub in self.epochs.keys() for sub in self.subjects):
            epochs = dict()
            for key in self.epochs.keys():
                epochs[key] = self.epochs[key][condition]
        else:
            raise(IndexError("Shouldn't be possible"))

        power = self.power[condition]
        sig = self.sig[condition]
        zscore = self.zscore[condition]
        conds = {condition: self.conds[condition]}

        return type(self)(self.root, self.task, self.units, conds, zscore, sig,
                        power, epochs, self.names)

    def plot_groups_on_average(self, groups: list[list[int]] = None,
                               colors: list[str] = ('red', 'green', 'blue')):
        assert hasattr(self, 'SM')
        cond = list(list(self.epochs.values())[0].keys())[0]
        plot_data = self.get_condition(cond)
        subjects = [v for v in plot_data.epochs.values() if v]
        if groups is None:
            groups = [self.SM, self.AUD, self.PROD]
        itergroup = (g for g in groups)
        if isinstance(colors, tuple):
            colors = list(colors)
        brain = plot_on_average(subjects, picks=next(itergroup),
                                color=colors.pop(0))
        for g, c in zip(itergroup, colors):
            plot_on_average(subjects, picks=g, color=c, fig=brain)
        return brain

    def get_training_data(self, dtype, conds, idx=None):
        assert dtype in ['power', 'zscore', 'sig']
        data = getattr(self, dtype)
        if isinstance(data, dict):
            data = list(data.get(cond) for cond in conds)
            data = np.concatenate(data, axis=1)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise ValueError(f"Invalid type for data {type(data)}")

        if idx is None:
            idx = self.sig_chans
        data = data[idx]
        return data

    def nmf(self, dtype: str = 'sig', n_components: int = 4,
            idx: list[int] = None,
            conds: list[str] = ('aud_ls', 'go_ls', 'resp'),
            plot: bool = True, plot_dtype: str = None):
        data = self.get_training_data(dtype, conds, idx)
        data = data - np.min(data)
        nmf = NMF(n_components=n_components, init='nndsvda', random_state=0,
                  solver='mu', max_iter=1000, tol=1e-6)
        W = nmf.fit_transform(data)
        H = nmf.components_
        if plot:
            if plot_dtype is None:
                plot_dtype = dtype
            plot_data = self.get_training_data(plot_dtype, conds, idx)
            plot_weight_dist(plot_data, W)
            labels = np.argmax(W, axis=1)
            groups = [[idx[i] for i in np.where(labels == j)[0]]
                      for j in range(n_components)]
            self.plot_groups_on_average(groups, colors=[
                'blue', 'orange', 'green', 'red'])
        return W, H


if __name__ == "__main__":
    data = GroupData()
    # W, H = subject_data.nmf(idx=subject_data.SM, conds=('aud_ls', 'resp'), plot_dtype='zscore')
    # W, H = subject_data.nmf(n_components=3, plot_dtype='zscore')
    # new_groups = subject_data.copy()
