import os
import mne
import numpy as np
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from ieeg.viz.mri import plot_on_average
from ieeg.calc.mat import LabeledArray, combine
from collections.abc import Sequence
from utils.mat_load import group_elecs, load_dict

import nimfa
from scipy import sparse


mne.set_log_level("ERROR")


class GroupData:

    @classmethod
    def from_intermediates(cls, task: str, root: PathLike,
                           conds: dict[str, Doubles] = None,
                           folder: str = 'stats', fdr=False,
                           pval: float = 0.05):
        layout = get_data(task, root=root)
        conds = cls._set_conditions(conds)
        sig = load_dict(layout, conds, "significance", True, folder)
        sig = combine(sig, (0, 2))
        data = dict(power=load_dict(layout, conds, "power", False, folder),
                    zscore=load_dict(layout, conds, "zscore", False, folder))
        data = combine(data, (1, 4))
        # subjects = tuple(data['power'].keys())
        out = cls(data, sig, fdr=fdr, pval=pval)
        # out.subjects = subjects
        out.task = task
        out._root = root
        return out

    def __init__(self, data: dict | LabeledArray,
                 mask: dict[str, np.ndarray] | LabeledArray = None,
                 categories: Sequence[str] = ('dtype', 'epoch', 'stim',
                                              'channel', 'trial', 'time'),
                 fdr: bool = False, pval: float = 0.05):
        self._set_data(data, '_data')
        self._categories = categories
        if mask is not None:
            self._set_data(mask, '_significance')
            if not ((self.sig == 0) | (self.sig == 1)).all():
                if 'epoch' in categories:
                    for i, arr in enumerate(self.sig):
                        self._significance[i] = self.correction(arr, fdr, pval)
                else:
                    self._significance = self.correction(self.sig, fdr, pval)
            keys = self._significance.labels
            if all(cond in keys[0] for cond in
                   ["aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm"]):

                self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
                    self._significance, keys[1], keys[0])
            else:
                self.sig_chans = self._find_sig_chans(self.sig)

    @staticmethod
    def correction(p_vals, fdr: bool, thresh: float):
        if fdr:
            p_vals = mne.stats.fdr_correction(p_vals)[1]
        return p_vals < thresh

    def _set_data(self, data: dict | LabeledArray, attr: str):
        if isinstance(data, dict):
            setattr(self, attr, LabeledArray.from_dict(data))
        elif isinstance(data, LabeledArray):
            setattr(self, attr, data)
        else:
            raise TypeError(f"input has to be dict or LabeledArray, not "
                            f"{type(data)}")

    @property
    def shape(self):
        return self._data.shape

    @property
    def keys(self):
        keys = self._data.labels
        return {self._categories[i]: tuple(k) for i, k in enumerate(keys)}

    @property
    def array(self):
        return self._data.__array__()

    @property
    def sig(self):
        if hasattr(self, '_significance'):
            return self._significance.__array__()

    @property
    def subjects(self):
        return set(f"{ch[:5]}" for ch in self.keys['channel'])

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

    def combine(self, levels: tuple[str, str]):
        assert all(lev in self._categories for lev in levels), "Invalid level"
        lev_nums = tuple(self._categories.index(lev) for lev in levels)
        new_data = self._data.combine(lev_nums)
        new_cats = list(self._categories)
        new_cats.pop(lev_nums[0])
        new_sig = None
        if not hasattr(self, '_significance'):
            pass
        elif all(self.keys[lev] in self._significance.labels for lev in levels):
            new_sig = self._significance.combine(lev_nums)
        return type(self)(new_data, new_sig, new_cats)

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

    def drop_nan(self, verbose: bool = False):
        # sig_chans = [self.keys['channel'][i] for i in self.sig_chans]
        out1 = None
        for i in self.sig_chans:
            ch = self.keys['channel'][i]
            out2 = None
            for j, st in enumerate(self.keys['stim']):
                temp = self._data[:, st, ch].dropna()
                temp = LabeledArray(temp[:, None, None], (temp.labels[0],)
                                    + ((st,),) + ((ch,),) + temp.labels[1:])
                if out2 is None:
                    out2 = temp
                else:
                    no_good = np.any(np.isnan(temp), axis=-2)
                    out2.append(temp[..., no_good == False, :], axis=1)
            if out1 is None:
                out1 = out2
            else:
                no_good = np.any(np.isnan(out2), axis=-2)
                out1.append(out2[..., no_good == False, :], axis=2)
        return out1

    def filter(self, item: str):
        """Filter data by key

        Takes the underlying self._data nested dictionary, finds the first
        level with a key that matches the item, and returns a new SubjectData
        object with the all other keys removed at that level. """

        new_categories = list(self._categories)

        def inner(data, lvl=0):
            if isinstance(data, LabeledArray):
                if item in data.keys():
                    new_categories.pop(lvl)
                    return data[item]
                else:
                    return {k: inner(v, lvl + 1) for k, v in data.items()}
            elif isinstance(data, np.ndarray):
                raise IndexError(f"'{item}' is not a valid key")
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

        if item in self._significance.keys():
            sig = inner(self._significance)
        else:
            sig = self._significance

        return type(self)(inner(self._data), sig, tuple(new_categories))

    def copy(self):
        return type(self)(self._data, self._significance, self._categories)

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
            if isinstance(obj, (int, float, bool, str)):
                return obj.__sizeof__()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
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

    def plot_groups_on_average(self, groups: list[list[int]] = None,
                               colors: list[str] = ('red', 'green', 'blue'),
                               **kwargs) -> mne.viz.Brain:
        if groups is None:
            assert hasattr(self, 'SM')
            groups = [self.SM, self.AUD, self.PROD]

        if isinstance(groups[0][0], int):
            groups = [[self.keys['channel'][idx] for idx in g] for g in groups]

        itergroup = (g for g in groups)
        if isinstance(colors, tuple):
            colors = list(colors)
        brain = plot_on_average(self.subjects, picks=next(itergroup),
                                color=colors.pop(0), **kwargs)
        for g, c in zip(itergroup, colors):
            plot_on_average(self.subjects, picks=g, color=c, fig=brain, **kwargs)
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
            idx = self.sig_chans.combine(('stim', 'trial'))
        if dtype in ['power', 'zscore']:
            data = self[dtype].combine(('stim', 'trial'))
            newconds = data.keys['epoch']
            gen = (np.nanmean(data[c].array[idx], axis=-2,
                              # where=self.significance[c][idx].astype(bool)
                              ) for c in conds)

        elif dtype == 'significance':
            data = self._significance
            newconds = self.keys['epoch']
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
                              objective='div', options=dict(flag=2))
        model()
        W = np.array(model.W)
        H = np.array(model.H)
        return W, H, model


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
    import cProfile
    import pstats

    fpath = os.path.expanduser("~/Box/CoganLab")
    with cProfile.Profile() as pr:
        sub = GroupData.from_intermediates("SentenceRep", fpath,
                                           folder='stats')

    stats = pstats.Stats(pr)
    stats.dump_stats('profile.stats')

    ##
    # power = sub['power'].combine(('stim', 'trial'))
    # resp = sub['resp']
    # resp_power = sub['power', 'resp']
    # # sm = sub.plot_groups_on_average([sub.SM], hemi='both')
    # # sm_wm = sub.plot_groups_on_average([sub.SM], hemi='both', rm_wm=False)
    #
    # ##
    # group = sub.SM
    # W, H, model = sub.nmf("significance", idx=group, n_components=3,
    #                        conds=('aud_lm', 'aud_ls', 'go_ls', 'resp'))
    # plot_data = sub.get_training_data("zscore", ("aud_ls", "go_ls"), group)
    # plot_data = np.hstack([plot_data[:, 0:175], plot_data[:, 200:400]])
    # plot_weight_dist(plot_data, W)
    # pred = np.argmax(W, axis=1)
    # groups = [[sub.keys['channel'][sub.SM[i]] for i in np.where(pred == j)[0]]
    #           for j in range(W.shape[1])]
    # fig1 = sub.plot_groups_on_average(groups,
    #                                 ['blue', 'orange', 'green', 'red'])
    #
    # ##
    # from MEPONMF.MEP_ONMF import DA
    # from MEPONMF.MEP_ONMF import ONMF_DA
    # import matplotlib.pyplot as plt
    # group = list(set(sub.SM + sub.PROD + sub.AUD))
    # group = sub.SM
    # k = 10
    # param = dict(tol=1e-14, alpha=1.0001,
    #            purturb=0.5, verbos=1, normalize=False)
    # model_data = sub.get_training_data("significance", ("aud_ls", "go_ls"), group)
    # # model_data = np.hstack([model_data[:, 0:175], model_data[:, 200:400]])
    # W, H, model = ONMF_DA.func(model_data, k=k, **param, auto_weighting=False)
    # model.plot_criticals(log=True)
    # plt.show()
    # ##
    # k = 2
    # W, H, model2 = ONMF_DA.func(model_data, k=k, **param, auto_weighting=True)
    # plot_weight_dist(plot_data, W)
    # # fig2 = data.plot_groups_on_average()
    #
    # ## plot conds
    # all_group = [data.AUD, data.PROD, data.SM]
    # plot_weight_dist(data.get_training_data("zscore", ("aud_ls",), all_group))