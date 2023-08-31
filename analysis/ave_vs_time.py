from analysis import GroupData
import os

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub_time = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
sub_ave = GroupData.from_intermediates("SentenceRep", fpath, folder='no_cluster')
