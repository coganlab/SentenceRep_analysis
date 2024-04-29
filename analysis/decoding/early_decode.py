from ieeg.viz.parula import mat_colors
import numpy as np
import os

from analysis.grouping import GroupData
from analysis.utils.plotting import plot_channels
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding import (Decoder, get_scores, plot_all_scores, plot_dist_bound, plot_dist)
from analysis.decoding.words import dict_to_structured_array, score

# %% Imports

def windowing_iterator(lst, window_size):
    extended_lst = lst + lst[:window_size-1]
    return [extended_lst[i:i+window_size] for i in range(len(lst))]

def random_iterator(lst, window_size, reps=1):
    out = []
    for _ in range(reps):
        for i, elem in enumerate(lst):
            out1 = None
            not_elem = lst[:i] + lst[i+1:]
            while out1 in out or out1 is None:
                idx = np.random.choice(len(not_elem), window_size-1, replace=False)
                out1 = tuple(sorted([elem] + [not_elem[j] for j in idx]))
            out.append(out1)
    return out

box = os.path.expanduser(os.path.join("~", "Box"))
fpath = os.path.join(box, "CoganLab")
subjects_dir = os.path.join(box, "ECoG_Recon")
sub = GroupData.from_intermediates(
    "SentenceRep", fpath, folder='stats', subjects_dir=subjects_dir)
all_data = []
idx = list(sub.SM)
subjects = np.unique(sub.keys['channel'][idx].astype('U5'))
subject = random_iterator(subjects.tolist(), 15, 1)
idxs = [[i for i in idx if any(s in sub.array.labels[3][i] for s in subj)] for subj in
        subject]
names = ["/".join(subj) for subj in subject]
colors = mat_colors
scores = {subj: None for subj in names}
conds = [['aud_ls', 'aud_lm'], ]
window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': 5,
                 'average_repetitions': False}

# %% Time Sliding decoding for word tokens

score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 1, sub, idxs,
      conds,
      window_kwargs, '../../test_scores.npy', scores,
      shuffle=False)

# %% Plot the results
import matplotlib.pyplot as plt
scores = {key: value for key, value in scores.items() if value is not None}
subj_score = {}
for subj in subjects:
    ins = []
    outs = []
    for key, values in scores.items():
        result = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
        if subj in key:
            ins.append(result)
        else:
            outs.append(result)
    subj_score[subj + "-aud_ls-aud_lm"] = (np.hstack(ins)[:, None]
                                           - np.hstack(outs)[:, :, None]).reshape((181, -1))
                                           # np.mean(np.array(ins + outs), axis=0)

pos = {subj: [i for i in idx if any(s in sub.array.labels[3][i] for s in subj)]
        for subj in subjects}
plot_all_scores(subj_score, conds, pos, colors, "Word Decoding")
plt.ylim((-0.15, 0.2))


all_scores = np.hstack(ins+outs)
plot_dist(all_scores.T, mode='std', color='red', times=(-0.4, 1.4))
# %%
for thresh in [0.31]:
    plots = {}; bad = {}; allscore = {}
    for key, values in scores.items():
        result = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
        if not np.any(thresh < np.mean(result[:45], axis=1)):
            plots[key] = result
        else:
            bad[key] = result
        allscore[key] = result

    pos = {n: i for n, i in zip(names, idxs) if any(n in p for p in plots.keys())}
    fig, axs = plot_all_scores(plots, conds, pos, colors, "Individual Decoding")
    # plt.legend('upper left')

    # bad.keys()
    # plt.figure()
    # all_dat = np.array(list(v[:30] for v in allscore.values()))
    # good_dat = np.array(list(v[:30] for v in plots.values()))
    # plt.hist(np.mean(all_dat, axis=2).flatten(), bins=100)
    # # plt.axvline(np.mean(all_dat), color='blue')
    # plt.hist(np.mean(good_dat, axis=2).flatten(), bins=100)
    # # plt.axvline(np.mean(good_dat), color='orange')
    # plt.title(f"Threshold: {thresh}")

# %%

new_plots = {k: v for i, (k, v) in enumerate(plots.items()) if i in [0, 13]}
pos = {n: i for n, i in zip(names, idxs) if any(n in p for p in new_plots.keys())}
fig, axs = plot_all_scores(plots, conds, pos, colors, "Individual Decoding")

    # # %% Time Sliding decoding for word tokens
    # idxs2 = [[i for i in idx if sub.array.labels[3][i][:5] not in [b[:5] for b in bad.keys()]]]
    # scores2 = {'All': None}
    # score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 30, sub, idxs2,
    #       conds,
    #       window_kwargs, '../../out_scores.npy', scores2,
    #       shuffle=False)
    #
    # scores2 = {key: value for key, value in scores2.items() if value is not None}
    # result = {}
    # for key, values in scores2.items():
    #     result[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
    #
    # fig, axs = plot_all_scores(result, conds, dict(All=idx), colors, "Word Decoding")

# # %% Plot the results
#     bad_subjects = [n[:5] for n in bad.keys()]
#     data = sub.array['zscore', 'aud_ls'].combine((0, 2))
#     idx = set(np.where([n[:5] not in bad_subjects for n in sub.array.labels[3]])[0])
#     bad_data = data[idx & sub.SM,]
#     plot_channels(bad_data, sub.signif["aud_ls", idx & sub.SM])
