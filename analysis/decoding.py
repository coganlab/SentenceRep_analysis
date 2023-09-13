# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

from analysis.grouping import GroupData
from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')

# %% Create training set

trials = 9
conds = ['aud_ls', 'aud_lm', 'aud_jl']
idx = sub.AUD
reduced = sub[:, conds, :, idx, :, 50:125]
reduced.array = reduced.array.dropna()
reduced.smotify_trials()
# reduced = reduced.nan_common_denom(True, trials, True)
comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'trial'))
train = comb['zscore'].array.combine((0, 2)).dropna()
cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
labels = np.array([cats[k.split('-')[1]] for k in train.labels[0]])
# train.size

# %% decoder
folds = 5
decoder = PcaLdaClassification()
cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=10)

scores = []
acc = []
rep = 0
best_mat = None
for f, (train_idx, test_idx) in enumerate(cv.split(train, labels)):
    print("Fold {} of {}".format(f + 1 - rep * folds, folds))
    decoder.fit(train[train_idx], labels[train_idx])
    # pred = decoder.predict(train[test_idx])
    # mat = confusion_matrix(labels[test_idx], pred) / (len(pred) // 4)
    scores.append(decoder.score(train[test_idx], labels[test_idx]))
    acc.append((decoder, train_idx, test_idx))
    # this_acc = np.sum(mat[np.eye(4).astype(bool)]) / np.sum(mat)
    # acc.append(this_acc)
    # if this_acc == max(acc):
    #     best_mat = mat
    if f - rep * folds == folds - 1:
        rep += 1

best = acc[scores.index(max(scores))]

# %%
disp = ConfusionMatrixDisplay.from_estimator(
    best[0].model, train[best[2]], labels[best[2]],
    display_labels=list(cats.keys()), normalize='true')
