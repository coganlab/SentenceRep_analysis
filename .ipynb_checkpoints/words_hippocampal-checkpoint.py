# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import sys; print('Python %s on %s' %(sys.version, sys.platform))
sys.path.extend('/Users/yuchaowang/Documents/git/SentenceRep_analysis' )

import numpy as np
import os

if LAB_root is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        else:  # mac
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                    "CoganLab")

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, get_scores, plot_all_scores, plot_dist_bound)


def dict_to_structured_array(dict_matrices, filename='structured_array.npy'):
    # Get the keys and shapes
    keys = list(dict_matrices.keys())
    shape = dict_matrices[keys[0]].shape

    # Create a data type for the structured array
    dt = np.dtype([(key, dict_matrices[key].dtype, shape) for key in keys])

    # Create the structured array
    structured_array = np.zeros((1,), dtype=dt)

    # Fill the structured array
    for key in keys:
        structured_array[key] = dict_matrices[key]

    # Save the structured array to a file
    np.save(filename, structured_array)


def score(categories, test_size, method, n_splits, n_repeats, sub, idxs,
          conds, window_kwargs, scores_dict, shuffle=False):
    decoder = Decoder(categories, test_size, method, n_splits=n_splits, n_repeats=n_repeats)
    names = list(scores_dict.keys())
    while len(scores_dict) > 0:
        scores_dict.popitem()
    for key, values in get_scores(sub, decoder, idxs, conds, names, shuffle=shuffle, **window_kwargs):
        print(key)
        scores_dict[key] = values
    return scores_dict


#if __name__ == '__main__':

# %% Imports
box = os.path.expanduser(os.path.join("~","Box"))
fpath = os.path.join(box, "CoganLab")
subjects_dir = os.path.join(box, "ECoG_Recon")
sub = GroupData.from_intermediates(
    "SentenceRep", fpath, folder='stats', subjects_dir=subjects_dir)
all_data = []
colors = [[0, 1, 0]]
scores = {'Auditory': None}
idxs = [sub.AUD]
idxs = [list(idx) for idx in idxs]
names = list(scores.keys())
conds = [['aud_ls', 'aud_lm']]
window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                'average_repetitions': False}

# %% Time Sliding decoding for word tokens

scores = score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 1, sub, idxs, conds,
                            window_kwargs, scores,
                            shuffle=False)
dict_to_structured_array(scores, 'true_scores.npy')

# %% Plotting
data_dir = ''
true_scores = np.load(data_dir + 'true_scores.npy', allow_pickle=True)[0]
true_scores = {name: true_scores[name] for name in true_scores.dtype.names}

plots = {}
for key, values in scores.items():
    if values is None:
        continue
    plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
fig, axs = plot_all_scores(plots, conds, {n: i for n, i in zip(names, idxs)}, colors, "Word Decoding")

for ax in fig.axes:
    ax.axhline(0.25, color='k', linestyle='--')
