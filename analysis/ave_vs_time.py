from analysis.grouping import GroupData
from analysis.utils.plotting import plot_channels
import os

# %% Imports

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_old')

# %% plotting
# x = sub["zscore", "aud_ls", :, sub.AUD]
# x.smotify_trials()
# y = np.mean(x.combine(('stim', 'trial')).array,axis=0)
# plot_dist(y.__array__(), times=(-0.5, 1.5))

# %% Plot all channels
x = sub.array["zscore", "aud_ls", :, sub.AUD]
x.labels.insert(0, x.labels.pop(1))
x = x.combine((1, 2))
fig = plot_channels(x)
