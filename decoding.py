# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

# %% Imports
from analysis import SubjectData
import os
from ieeg.calc.mat import concatenate_arrays

fpath = os.path.expanduser("~/Box/CoganLab")
sub = SubjectData.from_intermediates("SentenceRep", fpath)
pow = sub['power']

# %% Create training set

conds=('aud_lm', 'aud_ls', 'go_ls', 'resp')
idx = sub.sig_chans
train = concatenate_arrays([pow[c].array[idx] for c in conds], axis=-1)

