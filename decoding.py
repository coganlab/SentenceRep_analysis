# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from analysis import SubjectData
import os
from ieeg.calc.mat import concatenate_arrays, ArrayDict

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

class NeuralSignalDecoder:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()

    def train(self, X_train, y_train):
        """
        Train the decoder with the given training data.

        X_train: numpy array
            Input features (neural signals) for training, shape (n_samples, n_features).
        y_train: numpy array
            Target labels (words) for training, shape (n_samples,).
        """
        self.lda.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict the word labels for the given test data.

        X_test: numpy array
            Input features (neural signals) for testing, shape (n_samples, n_features).

        Returns:
        predictions: numpy array
            Predicted word labels for the test data, shape (n_samples,).
        """
        predictions = self.lda.predict(X_test)
        return predictions

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = SubjectData.from_intermediates("SentenceRep", fpath)
pow = sub['power']
# resp = sub['resp']

# %% Create training set

conds = ('aud_lm', 'aud_ls', 'aud_jl')
exclude = tuple(k for k in pow.keys['condition'] if k not in conds)
idx = sub.sig_chans
comb = sub.copy()['power']
comb._data = pow._data.combine_dims((1, 3))
train = concatenate_arrays([comb[c].array[idx] for c in conds], axis=-1)
train = train.swapaxes(0, 1)
new = ArrayDict(**comb._data)
for k in exclude:
    new.pop(k, None)
labels = [k.split('-')[0] for k in new.all_keys[1]]
# x = sub[conds]

clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

time_decod = SlidingEstimator(clf, n_jobs=-1, scoring="roc_auc", verbose=True)
# here we use cv=3 just for speed
# give y the
scores = cross_val_multiscore(time_decod, train, labels, cv=5, n_jobs=-1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)