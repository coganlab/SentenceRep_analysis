import os
import mne
import numpy as np
from functools import cache
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from ieeg.viz.utils import plot_weight_dist
from ieeg.viz.mri import get_sub_dir, plot_on_average
from plotting import compare_subjects, plot_clustering
from utils.mat_load import load_intermediates, group_elecs
from copy import deepcopy
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import nimfa


class GroupData:
    """Class for loading and analyzing group data"""
    def __init__(self, d_data: PathLike = None, task: str = "SentenceRep",
                 units: str = "uV", conditions: dict[str, Doubles] = None,
                 _zscore=None, _significance=None, _power=None, _subjects=None,
                 _names=None):
        mne.set_log_level("ERROR")
        self._root: PathLike = self._set_root(d_data)
        self.task: str = task
        self.units: str = units
        self.conds: dict = self._set_conditions(conditions)

        if any(at is None for at in
               [_significance, _power, _zscore, _names, _subjects]):
            layout = get_data(self.task, root=self._root)
            if _significance is None or _names is None or _subjects is None:
                epochs, sig, self._names = load_intermediates(
                    layout, self.conds, "significance")
                subjects = list(set(n[:5] for n in self._names))
                subjects.sort()
                self.subjects: dict = {s: list(epochs[s].values())[0].info
                                       for s in subjects}
                del epochs
            else:
                sig = _significance
                self._names: list = _names
                self.subjects: dict = _subjects
        else:
            layout = None
            sig = _significance
            self._names: list = _names
            self.subjects: dict = _subjects

        power = self._load_intermediates(_power, layout, "power")
        zscore = self._load_intermediates(_zscore, layout, "zscore")

        if sig is not False:
            if all(cond in self.conds.keys() for cond in
                   ["aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm"]):
                self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
                    sig, self._names, self.conds)
            else:
                self.sig_chans = self._find_sig_chans(sig)

        self.subjects_dir = self._set_subjects_dir()
        self._data = OrderedDict(significance=sig, power=power, zscore=zscore)
        self.shape: tuple = self.asarray().shape

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

    @property
    def array(self):
        return self.__array__()

    def _load_intermediates(self, input_data: dict, layout, attr: str):
        if input_data is None:
            _, val, _ = load_intermediates(layout, self.conds, attr)
        else:
            val = input_data
        return val

    def __repr__(self):
        size = self.__sizeof__()
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if size < 1024.0 or unit == 'PiB':
                break
            size /= 1024.0
        return f"GroupData({self.task}, {len(self.subjects)} subjects, " \
               f"conditions: {list(self.conds.keys())} ~{size:.{1}f} {unit})"

    def __sizeof__(self):
        def inner(obj):
            if isinstance(obj, (int, float, bool, str, np.ndarray)):
                return obj.__sizeof__()
            elif isinstance(obj, (tuple, list, set, frozenset)):
                return sum(inner(i) for i in obj)
            elif isinstance(obj, dict):
                return sum(inner(k) + inner(v) for k, v in obj.items())
            elif hasattr(obj, '__dict__'):
                return inner(vars(obj))
            else:
                return obj.__sizeof__()

        return inner(self)

    def __getitem__(self, item: str | int | slice | list | tuple, axis=-2):

        if isinstance(item, (list, tuple, np.ndarray, int)):
            return np.take(self.array, item, axis=axis)
        elif item in self.subjects:
            return self.get_subject(item)
        elif item in self.conds.keys():
            return self.get_condition(item)
        elif item in self._data.keys():
            return self.get_data(item)
        elif item in self._names:
            return self.get_elec(item)
        else:
            raise IndexError(f"{item} not in {self}")

    def __len__(self):
        return self.shape[-2]

    def __iter__(self):
        return np.ndarray.__iter__(self.__array__())

    @cache
    def __array__(self):
        # recurses through the data dictionaries to get the data
        def inner(data):
            if isinstance(data, dict):
                return np.array([inner(d) for d in data.values() if d
                                 is not False])
            else:
                return data
        return np.squeeze(inner(self._data))

    def asarray(self):
        return self.__array__()

    def copy(self):
        return deepcopy(self)

    def get_elec(self, elec: str | int):
        if isinstance(elec, str):
            elec = self._names.index(elec)
        return self.__array__()[elec]

    def get_data(self, data: str):
        out_data = dict()
        for k in self._data.keys():
            if k != data:
                out_data["_" + k] = False
            else:
                out_data["_" + k] = self._data[k]
        return type(self)(self._root, self.task, self.units, self.conds,
                          _subjects=self.subjects, _names=self._names,
                          **out_data)

    def get_subject(self, sub_id: str):
        assert sub_id in self.subjects.keys()
        idx = [i for i, n in enumerate(self._names) if sub_id in n]
        names = [self._names[i] for i in idx]

        def process_data(d, i):
            if isinstance(d, dict):
                nd = dict()
                for k, v in d.items():
                    nd[k] = process_data(v, i)
                return nd
            else:
                return d[i, :]

        nd = process_data(self._data, idx)

        for k in nd.keys():
            nd["_" + k] = nd.pop(k, False)

        return type(self)(self._root, self.task, self.units, self.conds,
                          _subjects={sub_id: self.subjects[sub_id]},
                          _names=names, **nd)

    def get_condition(self, condition: str):
        assert condition in self.conds.keys()
        out = dict()
        for k, v in self._data.items():
            if isinstance(v, dict):
                out["_" + k] = v[condition]
            else:
                out["_" + k] = v
        conds = {condition: self.conds[condition]}

        return type(self)(self._root, self.task, self.units, conds, **out,
                          _subjects=self.subjects, _names=self._names)

    def plot_groups_on_average(self, groups: list[list[int]] = None,
                               colors: list[str] = ('red', 'green', 'blue'),
                               rm_wm: bool = True):
        subjects = [v for v in self.subjects.values() if v]
        if groups is None:
            assert hasattr(self, 'SM')
            groups = [self.SM, self.AUD, self.PROD]

        if isinstance(groups[0][0], int):
            groups = [[self._names[idx] for idx in g] for g in groups]

        itergroup = (g for g in groups)
        if isinstance(colors, tuple):
            colors = list(colors)
        brain = plot_on_average(subjects, picks=next(itergroup),
                                color=colors.pop(0), rm_wm=rm_wm)
        for g, c in zip(itergroup, colors):
            plot_on_average(subjects, picks=g, color=c, fig=brain, rm_wm=rm_wm)
        return brain

    def groups_from_weights(self, w: np.ndarray[float], idx: list[int] = None):
        if idx is None:
            idx = self.sig_chans
        labels = np.argmax(w, axis=1)
        groups = [[idx[i] for i in np.where(labels == j)[0]]
                  for j in range(w.shape[1])]
        return groups

    def get_training_data(self, dtype: str,
                          conds: list[str] | str = ('aud_ls', 'go_ls'),
                          idx=None) -> np.ndarray:
        assert dtype in ['power', 'zscore', 'significance']
        data = self[dtype]
        if isinstance(conds, str):
            conds = [conds]
        if idx is None:
            idx = self.sig_chans
        assert all(cond in data.conds for cond in conds)
        return np.concatenate([data[c][idx] for c in conds], axis=-1)

    def nmf(self, dtype: str = 'significance', n_components: int = 4,
            idx: list[int] = None,
            conds: list[str] = ('aud_ls', 'go_ls', 'resp')) -> Doubles:
        data = self.get_training_data(dtype, conds, idx)
        data = data - np.min(data)
        model = nimfa.Bmf(data, seed="nndsvd", rank=n_components, max_iter=100000,
                        lambda_w=1.01, lambda_h=1.01, options=dict(flag=2))
        model()
        # nmf = NMF(n_components=n_components, init='nndsvda', random_state=0,
        #           solver='mu', max_iter=1000, tol=1e-6)
        # W = nmf.fit_transform(data)
        # H = nmf.components_
        return np.array(model.W), np.array(model.H), model


def convert_matrix(matrix):
    max_val = max(matrix) + 1
    result = [[int(i == j) for j in range(max_val)] for i in matrix]
    return result


if __name__ == "__main__":
    data = GroupData()
    # for i in range(1):
    #     resp = data['resp']
    #     resp_d3 = resp['D0003']
    #     resp_d3_sig = resp_d3['significance']
    #     D3 = data['D0003']
    #     power = data['power']
    ##
    W, H, model = data.nmf(idx=data.SM, conds=('aud_lm', 'aud_ls', 'go_ls', 'resp'))
    plot_data = data.get_training_data("zscore", ("aud_ls", "go_ls"), data.SM)
    plot_weight_dist(plot_data, W)
    pred = np.argmax(W, axis=1)
    groups = [[data._names[data.SM[i]] for i in np.where(pred == j)[0]]
              for j in range(W.shape[1])]
    fig1 = data.plot_groups_on_average(groups,
                                       ['blue', 'orange', 'green', 'red'])
    ##
    # labels = np.argmax(W, axis=1)
    # groups = [[data.SM[i] for i in np.where(labels == j)[0]]
    #           for j in range(W.shape[1])]
    # data.plot_groups_on_average(groups, ['blue', 'orange', 'green', 'red'])
    ##
    # all_group = data.SM + data.AUD + data.PROD
    # all_labels = np.concatenate([np.full(len(n), v) for n, v in zip(
    #     [data.SM, data.AUD, data.PROD], [0, 1, 2])])
    # W2, H2, model2 = data.nmf(idx=all_group, n_components=3,
    #                   conds=('aud_lm', 'aud_ls', 'go_ls', 'resp'))
    # plot_data2 = data.get_training_data("zscore", ("aud_ls", "go_ls"),
    #                                     idx=all_group)
    # plot_weight_dist(plot_data2, W2)
    # pred = np.argmax(W2, axis=1)
    # groups = [[data._names[all_group[i]] for i in np.where(pred == j)[0]]
    #           for j in range(W2.shape[1])]
    # fig1 = data.plot_groups_on_average(groups, ['blue', 'orange', 'green', 'red'])
    # plot_weight_dist(plot_data2, all_labels)
    # fig2 = data.plot_groups_on_average()
    # fig3 = data.plot_groups_on_average([data.SM], ['red'], rm_wm=False)
    # fig4 = data.plot_groups_on_average([data.AUD], ['green'], rm_wm=False)
