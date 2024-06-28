import os
import mne
import numpy as np
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from ieeg.viz.mri import plot_on_average
from ieeg.viz.ensemble import plot_dist
from ieeg.calc.mat import LabeledArray, combine
from collections.abc import Sequence
from analysis.utils.mat_load import load_dict
from ieeg.viz.mri import subject_to_info, gen_labels, get_sub, pick_no_wm
import matplotlib.pyplot as plt

import nimfa
from scipy import sparse


mne.set_log_level("ERROR")


class GroupData:
    array = LabeledArray([])
    signif = LabeledArray([])

    @classmethod
    def from_intermediates(cls, task: str, root: PathLike,
                           conds: dict[str, Doubles] = None,
                           folder: str = 'stats', **kwargs):
        layout = get_data(task, root=root)
        conds = cls._set_conditions(conds)
        sig = load_dict(layout, conds, "significance", True, folder)
        sig = combine(sig, (0, 2))
        pwr = load_dict(layout, conds, "power", False, folder)
        zscore = load_dict(layout, conds, "zscore", False, folder)
        data = combine(dict(power=pwr, zscore=zscore), (1, 4))
        # subjects = tuple(data['power'].keys())
        pvals = load_dict(layout, conds, "pval", True, folder)
        if pvals:
            kwargs['pvals'] = combine(pvals, (0, 2))

        out = cls(data, sig, **kwargs)
        # out.subjects = subjects
        out.task = task
        out._root = root
        return out

    def __init__(self, data: dict | LabeledArray,
                 mask: dict[str, np.ndarray] | LabeledArray = None,
                 categories: Sequence[str] = ('dtype', 'epoch', 'stim',
                                              'channel', 'trial', 'time'),
                 pvals: dict[str, np.ndarray] | LabeledArray = None,
                 fdr: bool = False, pval: float = 0.05,
                 wide: bool = False, subjects_dir: PathLike = None
                 , per_subject: bool = False):
        self._set_data(data, 'array')
        self._categories = categories
        self.subjects_dir = subjects_dir
        if mask is not None:
            self._set_data(mask, 'signif')
            if not ((self.signif == 0) | (self.signif == 1)).all():
                self.p_vals = self.signif.copy()
                if 'epoch' in categories:
                    for i, arr in enumerate(self.signif):
                        self.signif[i] = self.correction(arr, fdr, pval, per_subject)
                else:
                    self.signif = self.correction(self.signif, fdr, pval, per_subject)
            elif pvals:
                self._set_data(pvals, 'p_vals')
            keys = self.signif.labels
            if all(cond in keys[0] for cond in
                   ["aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "resp"]):

                self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
                    self.signif, keys[1], keys[0], wide=wide)
            else:
                self.sig_chans = self._find_sig_chans(self.signif)

    def correction(self, p_vals, fdr: bool, thresh: float,
                   per_subject: bool = False, verbose=False):

        if per_subject:
            sig = np.zeros_like(p_vals, dtype=bool)
            for sub in self.subjects:
                idx = self.keys['channel'].astype('U5') == sub
                if fdr:
                    temp = p_vals[idx]
                    sig[idx] = mne.stats.fdr_correction(temp, thresh)[0]
                    if verbose:
                        passed = np.logical_and(sig[:, 0], idx)
                        new_thsh = np.max(p_vals[(passed,)]) if np.any(passed) else 0
                        print(f"FDR correction applied, new threshold: {new_thsh}")
                else:
                    sig[idx] = p_vals[idx] < thresh
        elif fdr:
            sig = mne.stats.fdr_correction(p_vals, thresh)[0]
            if verbose:
                new_thresh = np.max(p_vals[(sig[:, 0],)])
                print(f"FDR correction applied, new threshold: {new_thresh}")
        else:
            sig = p_vals < thresh

        return sig

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
        return self.array.shape

    @property
    def keys(self):
        keys = self.array.labels
        return {self._categories[i]: k for i, k in enumerate(keys)}

    @property
    def subjects(self):
        return set(f"{ch[:5]}" for ch in self.keys['channel'])

    @property
    def grey_matter(self):
        if not hasattr(self, 'atlas'):
            self.atlas = ".a2009s"
        wm = get_grey_matter(self.subjects, self.subjects_dir, self.atlas)
        return {i for i, ch in enumerate(self.keys['channel']) if ch in wm}

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
        return np.where(np.any(np.atleast_2d(sig) == 1, axis=1))[0].tolist()

    def combine(self, levels: tuple[str, str]):
        assert all(lev in self._categories for lev in levels), "Invalid level"
        lev_nums = tuple(self._categories.index(lev) for lev in levels)
        new_data = self.array.combine(lev_nums)
        new_cats = list(self._categories)
        new_cats.pop(lev_nums[0])
        if not hasattr(self, 'signif'):
            new_sig = None
        elif np.any([np.array_equal(self.keys[levels[0]], l) for l in self.signif.labels])\
                and np.any([np.array_equal(self.keys[levels[1]], l) for l in self.signif.labels]):
            new_sig = self.signif.combine(lev_nums)
        else:
            new_sig = self.signif
        return type(self)(new_data, new_sig, new_cats)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        while len(item) < self.array.ndim:
            item += (slice(None),)

        sig = getattr(self, 'signif', LabeledArray([]))
        if sig.size > 0:
            sig_keys = []
            for i, key in enumerate(item):
                labels = self.array.labels[i]
                if np.any([np.array_equal(labels, l) for l in sig.labels]):
                    if isinstance(key, list):
                        key = tuple(key)
                    sig_keys.append(key)
            if sig_keys:
                sig = sig[tuple(sig_keys)]

        cats = tuple(self._categories[i] for i, key in enumerate(item) if
                   isinstance(key, (Sequence, slice)) and not isinstance(key, str))

        return type(self)(self.array[item], sig, cats)

    def nan_common_denom(self, sort: bool = True, min_trials: int = 0,
                         crop_trials: bool = True, verbose: bool = False):
        """Remove trials with NaNs from all channels"""
        trials_idx = self._categories.index('trial')
        ch_idx = self._categories.index('channel')
        others = [i for i in range(len(self._categories)) if ch_idx != i != trials_idx]
        isn = np.isnan(self.array)
        nan_trials = np.any(isn, axis=tuple(others))

        # Sort the trials by whether they are nan or not
        if sort:
            order = np.argsort(nan_trials, axis=1)
            old_shape = list(order.shape)
            new_shape = [1 if ch_idx != i != trials_idx else old_shape.pop(0)
                         for i in range(len(self._categories))]
            order = np.reshape(order, new_shape)
            data = np.take_along_axis(self.array.__array__(), order, axis=trials_idx)
            data = LabeledArray(data, self.array.labels.copy())
        else:
            data = self.array

        ch_tnum = self.shape[trials_idx] - np.sum(nan_trials, axis=1)
        ch_min = ch_tnum.min()
        if verbose:
            print(f"Lowest trials {ch_min} at "
                  f"{self.keys['channel'][ch_tnum.argmin()]}")

        ntrials = max(ch_min, min_trials)
        if ch_min < min_trials:
            # data = data.take(np.where(ch_tnum >= ntrials)[0], ch_idx)
            ch = np.array(self.keys['channel'])[ch_tnum < ntrials].tolist()
            if verbose:
                print(f"Channels excluded (too few trials): {ch}")

        # data = data.take(np.arange(ntrials), trials_idx)
        idx = [np.arange(ntrials) if i == trials_idx and crop_trials
               else np.arange(s) for i, s in enumerate(self.shape)]
        idx[ch_idx] = np.where([ch_tnum >= ntrials])[1]

        sig = getattr(self, '_significance', None)
        if sig is not None:
            sig_ch_idx = sig.labels.index(data.labels[ch_idx])
            sig = sig.take(idx[ch_idx], sig_ch_idx)

        return type(self)(data[np.ix_(*idx)], sig, self._categories)

    def plot_each(self, axis: tuple[int, ...] | int = None, n_cols: int = 10,
                  n_rows: int = 6, times: tuple[float, float] = None,
                  size: tuple[int, int] = (16, 12), metric: str = 'zscore'
                  ) -> list[plt.Figure]:
        """Plot each channel separately"""
        if axis is None:
            axis = tuple(self._categories.index('channel'))
        elif isinstance(axis, int):
            axis = (axis,)
        if times is None:
            times = (self.array.labels[-1][0], self.array.labels[-1][-1])
        if 'dtype' in self._categories:
            arr = self[metric].array.copy()
        else:
            arr = self.array.copy()

        while len(axis) > 1:
            arr = arr.combine(axis[:2])
            axis = tuple(i - 1 for i in axis[1:])

        per_fig = n_cols * n_rows
        chans = arr.labels[axis[-1]]
        numfigs = int(np.ceil(len(chans) / per_fig))
        figs = []
        indices = np.linspace(times[0], times[-1], arr.shape[-1])
        for i in range(numfigs):
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, frameon=False,
                                    figsize=size)
            for j, ch in enumerate(chans[i * per_fig:(i + 1) * per_fig]):
                ax = axs.flatten()[j]
                idx = tuple(slice(None) if k != axis[-1]
                            else ch for k in range(arr.ndim))
                plot_dist(arr[idx].__array__(), ax=ax, times=times)
                bvect = vect2ind(self.signif[ch].__array__())
                loc = [(indices[b[0]], indices[b[1]]) for b in bvect]
                ax.broken_barh(loc, (-0.3, 0.3), color='black')
                ax.set_title(ch)
            ymax = max(ax.get_ylim()[1] for ax in axs.flatten())
            ymin = min(ax.get_ylim()[0] for ax in axs.flatten())
            for ax in axs.flatten():
                ax.set_ylim(ymin, ymax)

            if i == numfigs - 1:
                while j + 1 < n_cols * n_rows:
                    j += 1
                    ax = axs.flatten()[j]
                    ax.axis("off")

            fig.tight_layout()
            figs.append(fig)
        return figs

    def filter(self, item: str):
        """Filter data by key

        Takes the underlying self.array nested dictionary, finds the first
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

        if hasattr(self, '_significance'):
            temp = new_categories.copy()
            if item in self.signif.keys():
                sig = inner(self.signif)
            else:
                sig = self.signif
            new_categories = temp
        else:
            sig = None

        return type(self)(inner(self.array), sig, tuple(new_categories))

    def copy(self):
        if hasattr(self, '_significance'):
            sig = self.signif
        else:
            sig = None
        return type(self)(self.array, sig, self._categories)

    def append(self, data):
        """Add entry to underlying data dictionary if other nested keys match

        Takes a nested dictionary and adds it to the underlying self.array
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

        inner(self.array, data.array)

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
        return self.array.__iter__()

    def plot_groups_on_average(self, groups: list[list[int]] = None,
                               colors: list[str] = ('red', 'green', 'blue'),
                               **kwargs) -> mne.viz.Brain:
        if groups is None:
            assert hasattr(self, 'SM')
            groups = [self.SM, self.AUD, self.PROD]
            groups = list(map(list, groups))

        if isinstance(groups[0][0], int):
            groups = [[self.keys['channel'][idx] for idx in g] for g in groups]

        itergroup = (g for g in groups)
        if isinstance(colors, tuple):
            colors = list(colors)
        if kwargs.get('fig') is None:
            kwargs['fig'] = plot_on_average(self.subjects, picks=next(itergroup),
                                    color=colors.pop(0), **kwargs)
        for g, c in zip(itergroup, colors):
            plot_on_average(self.subjects, picks=g, color=c, **kwargs)
        return kwargs['fig']

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
            data = self.signif
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
        all = self.array
        all["significance"] = self.signif
        out = dict()
        for k, v in all.items():
            if isinstance(v, dict):
                out["_" + k] = v[condition]
            else:
                out["_" + k] = v
        conds = {condition: self.conds[condition]}

        return type(self)(self._root, self.task, self.units, conds, **out,
                          _subjects=self.subjects, _names=self._names)


def vect2ind(bvect):
    # Find the indices where the binary vector changes
    change_indices = np.where(np.diff(bvect) != 0)[0] + 1

    # If the binary vector starts with a non-zero value, add 0 at the beginning
    if bvect[0] != 0:
        change_indices = np.insert(change_indices, 0, 0)

    # If the binary vector ends with a non-zero value, add the length of the vector at the end
    if bvect[-1] != 0:
        change_indices = np.append(change_indices, len(bvect))

    # Reshape the array into a list of tuples
    tuple_list = [(start, end) for start, end in
                  zip(change_indices[::2], change_indices[1::2])]
    return tuple_list


def convert_matrix(matrix):
    max_val = max(matrix) + 1
    result = [[int(i == j) for j in range(max_val)] for i in matrix]
    return result


def sparse_matrix(ndarray_with_nan: np.ndarray) -> sparse.spmatrix:
    non_nan_values = ndarray_with_nan[~np.isnan(ndarray_with_nan)]
    rows, cols = np.where(~np.isnan(ndarray_with_nan))
    return sparse.csr_matrix((non_nan_values, (rows, cols)),
                             shape=ndarray_with_nan.shape).tolil()


def group_elecs(all_sig: dict[str, np.ndarray] | LabeledArray, names: list[str],
                conds: tuple[str], wide: bool = False
                ) -> (set[int], set[int], set[int], set[int]):
    sig_chans = set()
    AUD = set()
    SM = set()
    PROD = set()
    for i, name in enumerate(names):
        for cond in conds:
            idx = i
            if wide:
                t_idx = slice(None)
            elif np.squeeze(all_sig).ndim == 2:
                t_idx = None
            elif cond in ["aud_ls", "aud_lm", "aud_jl"]:
                t_idx = slice(50, 100)
            else:
                t_idx = slice(75, 125)

            if np.any(all_sig[cond, idx, t_idx] == 1):
                sig_chans |= {i}
                break

        if all_sig.ndim == 2:
            aud_slice = None
            go_slice = None
        elif all_sig.shape[2] == 1:
            aud_slice = slice(None)
            go_slice = slice(None)
        elif wide:
            aud_slice = slice(50, 175)
            go_slice = slice(50, None)
        else:
            aud_slice = slice(50, 100)
            go_slice = slice(75, 125)

        audls_is = np.any(all_sig['aud_ls', idx, aud_slice] == 1)
        audlm_is = np.any(all_sig['aud_lm', idx, aud_slice] == 1)
        audjl_is = np.any(all_sig['aud_jl', idx, aud_slice] == 1)
        mime_is = np.any(all_sig['go_lm', idx, go_slice] == 1)
        resp_is = np.any(all_sig['resp', idx, go_slice] == 1)
        speak_is = np.any(all_sig['go_ls', idx, go_slice] == 1)

        if audls_is and audlm_is and mime_is and (speak_is or resp_is):
            SM |= {i}
        elif audls_is and audlm_is and audjl_is:
            AUD |= {i}
        elif mime_is and (speak_is or resp_is):
            PROD |= {i}

    return AUD, SM, PROD, sig_chans


def get_grey_matter(subjects: Sequence[str], subjects_dir: str = None, atlas: str = ".a2009s") -> set[str]:
    grey_matter = set()
    for i, subj in enumerate(subjects):
        info = subject_to_info(get_sub(subj), subjects_dir=subjects_dir)
        parcel_label = gen_labels(info, get_sub(subj), subjects_dir, atlas)
        subj_grey_matter = pick_no_wm(info.ch_names, parcel_label)

        # get the indices of channels in info that are not in grey_matter
        grey_matter |= {f"{subj}-{ch}" for ch in info.ch_names
                        if ch in subj_grey_matter}
    return grey_matter


if __name__ == "__main__":
    from scipy import stats as st
    fpath = os.path.expanduser("~/Box/CoganLab")
    sub = GroupData.from_intermediates("SentenceRep", fpath,
                                           folder='stats')
    conds = {"resp": (-1, 1),
             "aud_ls": (-0.5, 1.5),
             "aud_lm": (-0.5, 1.5),
             "aud_jl": (-0.5, 1.5),
             "go_ls": (-0.5, 1.5),
             "go_lm": (-0.5, 1.5),
             "go_jl": (-0.5, 1.5)}

    power = sub['power']
    # zscore = np.nanmean(sub['zscore'].__array__(), axis=(-4, -2))
    pval = np.where(sub.p_vals > 0.9999, 0.9999, sub.p_vals)

    # pval[pval<0.0001] = 0.0001
    zscore = LabeledArray(st.norm.ppf(1 - pval), sub.p_vals.labels)
    sig = sub.signif

    #
    # cond = 'aud_ls'
    # # arr = np.nanmean(power.array[cond].__array__(), axis=(-4, -2))
    # # arr = sub.signif[cond].__array__()
    # arr = np.nanmean(zscore.array[cond].__array__(), axis=(-4, -2))
    # for subj in list(sub.subjects)[:10]:
    #     SUB = [s for s in sub.sig_chans if subj in sub.keys['channel'][s]]
    #     plot_dist(arr[SUB], times=conds[cond], label=subj)
    # plt.legend()
    ##
    cond = 'aud_ls'
    # arr = np.nanmean(power.array[cond].__array__(), axis=(-4, -2))
    # arr = sub.signif[cond].__array__()
    arr = zscore[cond].__array__()
    fig = plt.figure()
    ax = fig.gca()
    plot_dist(arr[list(sub.AUD)], times=conds[cond], label='AUD',
              color='green', ax=ax)
    plot_dist(arr[list(sub.SM)], times=conds[cond], label='SM',
              color='red', ax=ax)
    plot_dist(arr[list(sub.PROD)], times=conds[cond], label='PROD',
              color='blue', ax=ax)
    plt.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Z-Score (V)")
    plt.title('Response')
    # plt.ylim(-0.1, 0.9)
    # plt.savefig(cond+'.svg', dpi=300)
    #
    ##
    fig = sub.plot_groups_on_average(rm_wm=False)
    fig.save_image('ALL.png')
