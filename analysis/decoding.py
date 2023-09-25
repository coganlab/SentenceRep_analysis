# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt

from analysis.grouping import GroupData
from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification
import mne


class Decoder(PcaLdaClassification):
    def cv_cm(self, train: np.ndarray, labels: np.ndarray, kfolds: int = 5,
              repeats: int = 10, normalize: str = 'true'):
        cv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats)
        rep = 0
        n_cats = len(set(labels))
        mats = np.zeros((repeats, kfolds, n_cats, n_cats))
        for f, (train_idx, test_idx) in enumerate(cv.split(train, labels)):
            fold = f + 1 - rep * kfolds
            print("Fold {} of {}".format(fold, kfolds))
            self.fit(train[train_idx], labels[train_idx])
            pred = self.predict(train[test_idx])
            mats[rep, fold-1] = confusion_matrix(labels[test_idx], pred)
            if f - rep * kfolds == kfolds - 1:
                rep += 1
        mats = np.mean(np.sum(mats, axis=1, keepdims=True), axis=(0, 1))
        if normalize == 'true':
            return mats / np.sum(mats, axis=0)
        elif normalize == 'pred':
            return mats / np.sum(mats, axis=1)
        elif normalize == 'all':
            return mats / np.sum(mats)
        else:
            return mats



# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')


# %%
scores = []
plt_labels = []
modes = [3,4,5,6,7,8,9,10,11,12]
         #,13,14,15,"smote-17"]
for i in modes:
    # Arrange
    conds = ['aud_ls', 'aud_lm', 'aud_jl']
    idx = sub.AUD
    reduced = sub[:, conds, :, idx, :, 50:125]
    reduced.array = reduced.array.dropna()
    if i != "smote-17":
        reduced = reduced.nan_common_denom(True, max(modes), True)[:, :, :, :, range(i), :]
    else:
        reduced.smotify_trials()
    comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'trial'))
    # concatenate channels across time as feature vector
    train = comb['zscore'].array.combine((0, 2)).dropna()
    cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
    labels = np.array([cats[k.split('-')[1]] for k in train.labels[0]])

    # Decoding
    kfolds = 5
    repeats = 10
    cv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats)
    scorer = make_scorer(balanced_accuracy_score)
    score = mne.decoding.cross_val_multiscore(
        PcaLdaClassification(), train, labels, verbose=10, cv=cv,
        scoring=scorer, n_jobs=1)
    scores.append(np.mean(np.reshape(score, (repeats, kfolds)), axis=1))
    plt_labels.append(f"{i}\n({reduced.shape[3]})")
plt.boxplot(scores)
plt.ylim(0, 1)
plt.xticks(list(range(1, len(modes)+1)), plt_labels)
plt.title("Balanced Accuracy")
plt.xlabel("Trials\n(channels)")
plt.tight_layout()
plt.show()
# %%

conds = ['aud_ls', 'aud_lm', 'aud_jl']
idx = sub.AUD
reduced = sub[:, conds, :, idx, :, 50:125]
reduced.array = reduced.array.dropna()
reduced = reduced.nan_common_denom(True, 12, True)
comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'trial'))
# concatenate channels across time as feature vector
train = comb['zscore'].array.combine((0, 2)).dropna()
cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
labels = np.array([cats[k.split('-')[1]] for k in train.labels[0]])
matrix = Decoder().cv_cm(train, labels, repeats=20)
fig = ConfusionMatrixDisplay(matrix, display_labels=cats.keys())
fig.plot(values_format='.3g')
