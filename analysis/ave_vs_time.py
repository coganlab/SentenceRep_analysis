from analysis.grouping import GroupData
from analysis.utils.plotting import plot_channels
import os

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')
fig1 = sub.plot_groups_on_average(rm_wm=False)
sub_ave = GroupData.from_intermediates("SentenceRep", fpath, folder='no_cluster', fdr=True)
fig2 = sub_ave.plot_groups_on_average(rm_wm=False)
sub_ave_no_fdr = GroupData.from_intermediates("SentenceRep", fpath, folder='no_cluster')
fig3 = sub_ave_no_fdr.plot_groups_on_average(rm_wm=False)
sub_wide = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt', wide=True)
fig4 = sub_wide.plot_groups_on_average(rm_wm=False)

# %% Plot all channels
x = sub.array["zscore", "aud_ls", :, sub.AUD]
x.labels.insert(0, x.labels.pop(1))
x = x.combine((1, 2))
fig = plot_channels(x)
