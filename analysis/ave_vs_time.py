from analysis.grouping import GroupData
from analysis.utils.plotting import plot_channels
import os

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')
sub_ave = GroupData.from_intermediates("SentenceRep", fpath, folder='no_cluster', fdr=True)
fig1 = sub.plot_groups_on_average()
fig2 = sub_ave.plot_groups_on_average()

# %% Plot all channels
x = sub.array["zscore", "aud_ls", :, sub.AUD]
x.labels.insert(0, x.labels.pop(1))
x = x.combine((1, 2))
fig = plot_channels(x)
