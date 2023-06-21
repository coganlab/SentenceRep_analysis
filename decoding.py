# Decoding script, takes a GroupData set and uses a recurrent neural network to decode trial conditions

# %% Imports
from analysis import SubjectData
import os

fpath = os.path.expanduser("~/Box/CoganLab")
sub = SubjectData.from_intermediates("SentenceRep", fpath)
