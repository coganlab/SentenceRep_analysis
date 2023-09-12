# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
from itertools import product

from analysis.grouping import GroupData
import os

from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')

# %% Create training set

reduced = sub.copy().nan_common_denom(min_trials=10, verbose=True)
idx = reduced.SM
# reduced.smotify_trials()
comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'channel'))
train = comb['power'].array.combine((0, 2)).dropna()
cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
labels = [cats[k.split('-')[0]] for k in train.labels[0]]

# %% decoder

decoder = PcaLdaClassification()
decoder.fit(train, labels)
pred = decoder.predict(train)
mat = confusion_matrix(labels, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=cats.keys())
disp.plot()
avg_percent = np.sum(mat[np.eye(4).astype(bool)]) / np.sum(mat)
