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

trials = 17
idx = sub.SM
reduced = sub.copy()[:, :, :, idx]
reduced.smotify_trials()
comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'channel'))
train = comb['power'].array.combine((0, 2)).dropna()
cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
labels = np.array([cats[k.split('-')[0]] for k in train.labels[0]])

# %% decoder
folds = 5
decoder = PcaLdaClassification()
cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=3)

scores = []
acc = []
rep = 0
for f, (train_idx, test_idx) in enumerate(cv.split(train, labels)):
    print("Fold {} of {}".format(f + 1 - rep * 5, folds))
    decoder.fit(train[train_idx], labels[train_idx])
    pred = decoder.predict(train[test_idx])
    mat = confusion_matrix(labels[test_idx], pred) / trials
    scores.append(decoder.score(train[test_idx], labels[test_idx]))
    acc.append(np.sum(mat[np.eye(4).astype(bool)]) / np.sum(mat))
    if f - rep * 5 == 4:
        rep += 1


# %%
disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=cats.keys())
disp.plot()
