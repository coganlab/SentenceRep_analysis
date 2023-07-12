from scipy.io import loadmat
from numpy import where, concatenate
from utils.calc import ArrayLike
from copy import deepcopy
import os
import numpy as np
from collections import OrderedDict
from functools import cache
from ieeg.viz.mri import get_sub_dir
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from utils.mat_load import load_intermediates, group_elecs
from ieeg.calc.mat import concatenate_arrays


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


def load_all(filename: str) -> tuple[dict, dict, dict, dict, dict, dict, list[dict]]:
    d = loadmat(filename, simplify_cells=True)
    t: dict = d['Task']
    z: dict = d['allSigZ']
    a: dict = d['sigMatA']
    # reduce indexing by 1 because python is 0-indexed
    for cond, epochs in d['sigChans'].items():
        for epoch, vals in epochs.items():
            d['sigChans'][cond][epoch]: ArrayLike = vals - 1
    sCh: dict = d['sigChans']
    sChL: dict = d['sigMatChansLoc']
    sChN: dict = d['sigMatChansName']
    sub: list = d['Subject']
    return t, z, a, sCh, sChL, sChN, sub


def group_elecs(sigA: dict[str, dict[str, ArrayLike]],
                sig_chans: dict[str, dict[str, list[int]]]
                ) -> tuple[list, list, list]:
    AUD = dict()
    for cond in ['LS', 'LM', 'JL']:
        condw = cond + 'words'
        elecs = where(sigA[condw]['AuditorywDelay'][:, 50:175] == 1)[0]
        AUD[cond] = set(elecs) & set(sig_chans[condw]['AuditorywDelay'])

    AUD1 = AUD['LS'] & AUD['LM']
    AUD2 = AUD1 & AUD['JL']
    PROD1 = set(sig_chans['LSwords']['DelaywGo']) & set(sig_chans['LMwords']['DelaywGo'])
    SM = list(AUD1 & PROD1)
    AUD = list(AUD2 - set(SM))
    PROD = list(PROD1 - set(SM))
    for group in [SM, AUD, PROD]:
        group.sort()
    return SM, AUD, PROD


def get_sigs(allsigZ: dict[str, dict[str, ArrayLike]], allsigA: dict[str, dict[str, ArrayLike]],
             sigChans: dict[str, dict[str, list[int]]], cond: str) -> tuple[dict[str, ArrayLike], dict[str, ArrayLike]]:
    out_sig = dict()
    for sig, metric in zip([allsigZ, allsigA], ['Z', 'A']):
        out_sig[metric] = dict()
        for group, idx in zip(['SM', 'AUD', 'PROD'], group_elecs(allsigA, sigChans)):
            blend = sig[cond]['AuditorywDelay'][idx, 150:175] / 2 + \
                    sig[cond]['DelaywGo'][idx, 0:25] / 2
            out_sig[metric][group] = concatenate((sig[cond]['AuditorywDelay'][idx, :150],
                                                  blend,
                                                  sig[cond]['DelaywGo'][idx, 25:]), axis=1)

    return out_sig['Z'], out_sig['A']


def get_bad_trials(subject: list[dict]):
    """Remove bad channels and trials from a dataset

    :param subject:
    :return:
    """
    for sub in subject:
        for trial in sub['Trials']:
            pass
    pass