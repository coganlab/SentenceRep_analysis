import os
import mne
import numpy as np
from functools import cache
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from ieeg.viz.utils import plot_weight_dist
from ieeg.viz.mri import get_sub_dir, plot_on_average
from ieeg.calc.mat import concatenate_arrays, ArrayDict
from collections.abc import Sequence
from plotting import compare_subjects, plot_clustering
from utils.mat_load import load_intermediates, group_elecs, load_dict
from copy import deepcopy
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import nimfa
from scipy import sparse


mne.set_log_level("ERROR")


class SubjectData:

    @classmethod
    def from_intermediates(cls, task: str, root: PathLike,
                           conds: dict[str, Doubles] = None):
        layout = get_data(task, root=root)
        conds = cls._set_conditions(conds)
        sig = ArrayDict(**load_dict(layout, conds, "significance"))
        sig = sig.combine_dims((0, 2))
        data = ArrayDict(power=load_dict(layout, conds, "power", False),
                         zscore=load_dict(layout, conds, "zscore", False))
        subjects = tuple(data['power'].keys())
        data = data.combine_dims((1, 4))
        out = cls(data, sig)
        out.subjects = subjects
        out.task = task
        out._root = root
        return out

    def __init__(self, data: dict, mask: dict[str, np.ndarray] = None,
                 categories: Sequence[str] = ('dtype', 'condition', 'stim',
                                              'channel', 'trial', 'time')):
        self._data = ArrayDict(**data)
        self._categories = categories
        if mask is not None:
            self.significance = ArrayDict(**mask)
            keys = self.significance.all_keys
            if all(cond in keys[0] for cond in
                   ["aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm"]):

                self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
                    self.significance, keys[1], keys[0])
            else:
                self.sig_chans = self._find_sig_chans(self.significance)

    @property
    def shape(self):
        return self._data.array.shape

    @property
    def keys(self):
        keys = self._data.all_keys
        return {self._categories[i]: tuple(k) for i, k in enumerate(keys)}

    @property
    def array(self):
        return self._data.array

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
    def _find_sig_chans(sig: np.ndarray) -> list[int]:
        return np.where(np.any(sig == 1, axis=1))[0].tolist()

    def __getitem__(self, item: str | Sequence[str]):
        if isinstance(item, str):
            return self.filter(item)
        elif not isinstance(item, Sequence):
            raise TypeError(f"Unexpected type: {type(item)}")
        elif all(isinstance(item[i], str) for i in range(len(item))):
            if len(item) == 1:
                return self[item[0]]
            keys = tuple(v for v in self.keys.values())
            level_groups = [[v for v in item if v in keys[i]] for i in range(len(keys))]
            level_groups = [g for g in level_groups if g]
            level_groups.reverse()

            # filter, append, then move up a level
            out = self.copy()
            for group in level_groups:
                this = out[group.pop(0)]
                while group:
                    this.append(out[group.pop(0)])
                out = this
            return out
        else:
            i = [isinstance(item[i], str) for i in range(len(item))
                 ].index(False)
            raise TypeError(f"Unexpected type: {type(item)}[{type(item[i])}]")

    def filter(self, item: str):
        """Filter data by key

        Takes the underlying self._data nested dictionary, finds the first
        level with a key that matches the item, and returns a new SubjectData
        object with the all other keys removed at that level. """

        new_categories = list(self._categories)

        def inner(data, lvl=0):
            if isinstance(data, dict):
                if item in data.keys():
                    new_categories.pop(lvl)
                    return data[item]
                else:
                    return {k: inner(v, lvl + 1) for k, v in data.items()}
            elif isinstance(data, np.ndarray):
                raise IndexError(f"'{item}' is not a valid key")
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

        if item in self._data.keys():
            sig = None
        else:
            sig = self.significance

        return type(self)(inner(self._data), sig, tuple(new_categories))

    def copy(self):
        return type(self)(self._data, self.significance, self._categories)

    def append(self, data):
        """Add entry to underlying data dictionary if other nested keys match

        Takes a nested dictionary and adds it to the underlying self._data
        dictionary. Appended data must have the same nested keys as the
        underlying data, except for the last level, which must be a single key
        that does not already exist in the underlying data."""

        if not isinstance(data, type(self)):
            raise TypeError(f"Unexpected type: {type(data)}")

        def inner(data, _data):
            if isinstance(data, dict):
                if all(k in _data.keys() for k in data.keys()):
                    for k, v in data.items():
                        inner(v, _data[k])
                elif len(data.keys()) == 1:
                    key = list(data.keys())[0]
                    _data[key] = data[key]
                else:
                    raise KeyError("Keys do not match")
            elif isinstance(data, np.ndarray):
                raise IndexError("Keys do not match")
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

        inner(self._data, data._data)

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

    def __repr__(self):
        size = self.__sizeof__()
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if size < 1024.0 or unit == 'PiB':
                break
            size /= 1024.0
        return f"GroupData({', '.join(f'{len(v)} {k}s' for k, v in self.keys.items())}) ~{size:.{1}f} {unit}"

    def __len__(self):
        return self.shape[-2]

    def __iter__(self):
        return self._data.__iter__()


class GroupData:
    """Class for loading and analyzing group data"""
    def __init__(self, d_data: PathLike = None, task: str = "SentenceRep",
                 units: str = "uV", conditions: dict[str, Doubles] = None,
                 _zscore=None, _significance=None, _power=None, _subjects=None,
                 _names=None):
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
        self._data = OrderedDict(power=power, zscore=zscore)
        self.significance = sig
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
            _, val, _ = load_intermediates(layout, self.conds, attr, False)
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
        elif hasattr(self, item):
            return getattr(self, item)
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
                return concatenate_arrays([inner(d) for d in data.values() if d
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
        out_data = dict(_significance=self.significance)
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
        nd["significance"] = process_data(self.significance, idx)
        for k in nd.keys():
            nd["_" + k] = nd.pop(k, False)

        return type(self)(self._root, self.task, self.units, self.conds,
                          _subjects={sub_id: self.subjects[sub_id]},
                          _names=names, **nd)

    def get_condition(self, condition: str):
        assert condition in self.conds.keys()
        all = self._data
        all["significance"] = self.significance
        out = dict()
        for k, v in all.items():
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
        if isinstance(conds, str):
            conds = [conds]
        if idx is None:
            idx = self.sig_chans
        if dtype in ['power', 'zscore']:
            data = self[dtype]
            newconds = data.conds
            gen = (np.nanmean(data[c][idx], axis=0,
                              where=self.significance[c][idx].astype(bool)
                              ) for c in conds)
        elif dtype == 'significance':
            data = self.significance
            newconds = self.conds
            gen = (data[c][idx] for c in conds)
        else:
            raise ValueError(f"{dtype} not in ['power', 'zscore', "
                             "'significance']")
        assert all(cond in newconds for cond in conds)
        return np.concatenate(list(gen), axis=-1)

    def nmf(self, dtype: str = 'significance', n_components: int = 4,
            idx: list[int] = None,
            conds: list[str] = ('aud_ls', 'go_ls', 'resp')) -> Doubles:
        data = self.get_training_data(dtype, conds, idx)
        data = data - np.min(data)
        if dtype == 'significance':
            model = nimfa.Bmf(data, seed="nndsvd", rank=n_components, max_iter=100000,
                            lambda_w=1.01, lambda_h=1.01, options=dict(flag=2))
        else:
            # data = sparse_matrix(data)
            model = nimfa.Nmf(data, seed="nndsvd",
                              rank=n_components,
                              max_iter=100000, update='euclidean',
                              objective='div', options=dict(flag=2)
                              )
        model()
        W = np.array(model.W)
        H = np.array(model.H)
        return W, H, model


def convert_matrix(matrix):
    max_val = max(matrix) + 1
    result = [[int(i == j) for j in range(max_val)] for i in matrix]
    return result


def sparse_matrix(ndarray_with_nan: np.ndarray) -> sparse.spmatrix:
    non_nan_values = ndarray_with_nan[~np.isnan(ndarray_with_nan)]
    rows, cols = np.where(~np.isnan(ndarray_with_nan))
    return sparse.csr_matrix((non_nan_values, (rows, cols)),
                             shape=ndarray_with_nan.shape).tolil()


if __name__ == "__main__":
    data = GroupData()
    # fpath = os.path.expanduser("~/Box/CoganLab")
    # sub = SubjectData.from_intermediates("SentenceRep", fpath)

    ##
    group = list(set(data.AUD + data.PROD + data.SM))

    W, H, model = data.nmf("significance", idx=group, n_components=3,
                           conds=('aud_lm', 'aud_ls', 'go_ls', 'resp'))
    plot_data = data.get_training_data("zscore", ("aud_ls", "go_ls"), group)
    plot_weight_dist(plot_data, W)
    pred = np.argmax(W, axis=1)
    groups = [[data._names[group[i]] for i in np.where(pred == j)[0]]
              for j in range(W.shape[1])]
    # fig1 = data.plot_groups_on_average(groups,
    #                                    ['blue', 'orange', 'green', 'red'])
    # fig2 = data.plot_groups_on_average()

    ## plot conds
    all_group = [data.AUD, data.PROD, data.SM]
    plot_weight_dist(data.get_training_data("zscore", ("aud_ls",), all_group))