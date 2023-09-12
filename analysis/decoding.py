# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
from itertools import product

from analysis.grouping import GroupData
import os
from ieeg.calc.mat import LabeledArray, concatenate_arrays

from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)

from IEEG_Pipelines.decoding.Neural_Decoding.decoders import PcaLdaClassification

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')
# pow = sub['power']
# resp = sub['resp']

# %% Create training set

conds = tuple(map("_".join, product(["aud", "go"], ["ls", "lm", "jl"])))
reduced = sub.copy()# .nan_common_denom(min_trials=5, verbose=True)
idx = reduced.SM
reduced.smotify_trials()
comb = reduced.combine(('stim', 'trial')).combine(('epoch', 'channel'))
train = comb['power'].combine((1, 3)).dropna()
# cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
# labels = [cats[k.split('-')[0]] for k in train.labels[0]]
# get_pre = lambda k: cats[k.split('-')[0]]
# dat = {c: (comb[c].array[idx], tuple(map(
#     get_pre, comb[c].array.labels[1]))) for c in conds}
# labels = [d[1] for d in dat.values()][1]
# train = concatenate_arrays([d[i] for d in dat.values() for i in range(len(d[0]))], axis=-1)


# %% decoder

decoder = PcaLdaClassification()
decoder.fit(train, labels)

# x = sub[conds]

# clf = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
#
# time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
# # here we use cv=3 just for speed
# # give y the
# scores = cross_val_multiscore(time_decod, dat[0][0], dat[0][1], cv=5, n_jobs=-1)
#
# # Mean scores across cross-validation splits
# scores = np.mean(scores, axis=0)
