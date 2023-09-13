from analysis.utils._mat_load_old import group_elecs
import matplotlib.pyplot as plt
import numpy as np

Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
SM, AUD, PROD = group_elecs(all_sigA, sig_chans)
winners, results, w_sav = np.load('data/nmf.npy', allow_pickle=True)
npSM = np.array(SM)

cond = 'LSwords'
SMresp = npSM[npSM < len(all_sigA[cond]['Response'])]
SMrespw = w_sav['SM'][npSM < len(all_sigA['LSwords']['Response'])]
names = ['Working Memory','Visual','Early Prod','Late Prod']
epochs = ['AuditorywDelay', 'DelaywGo', 'Response']

# %% find the latencies of the max values for each channel and plot the distribution for each group
epoch = epochs[1]

if epoch == 'Response':
    channels = SMresp
    groups = np.argmax(SMrespw, 1)
else:
    channels = SM
    groups = np.argmax(w_sav['SM'], 1)
sig = np.multiply(all_sigZ[cond][epoch][channels, 1:200], all_sigA[cond][epoch][channels, 1:200])
#sig = np.concatenate([sig, np.multiply(all_sigZ[cond][epochs[1]][channels, :], all_sigA[cond][epochs[1]][channels, :])],1)
latency = np.argmax(sig, 1)
max_vals = np.max(sig, 1)
latency = latency - 50
latency = latency / 100
colors = np.array([[0, 0, 0], [0.6, 0.3, 0], [.9, .9, 0], [1, 0.5, 0]])
fig, ax = plt.figure(), plt.subplot(111)
ax2=ax.twinx()
for i, name in enumerate(names):
    ax2.boxplot(latency[groups == i], positions=[i / 4 + 2], widths=0.25, whis=[10, 90],
               showfliers=False, vert=False, patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5))
    ax.scatter(latency[groups == i], max_vals[groups == i], color=colors[i], alpha=0.5, label=name)
ax.set_ylabel('Max Value (z-score)')
ax2.set_yticks([])
plt.title('Latency of Max Value for Each Cluster Channel')
ax.legend(names, loc='upper left')
ax2.set_ylim(ax.get_ylim())
# epoch = 'Delay'
ax.set_xlabel(epoch + ' Latency (s)')

# plt.title('Latency of Max Value for Each Channel')
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
# plot the distribution of latencies for each group with a horizontal boxplot
# for i, name in enumerate(names):
#     ax.boxplot(latency[groups == i], positions=[i/2+2], widths=0.25, whis=[10,90],
#                 showfliers=False, vert=False, patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5))
# plt.xlabel(epoch+' Latency (s)')
# plt.title('Latency of Max Value for Each Cluster Channel')
# ax2.yaxis.tick_left()
# ax.legend(names, loc='upper left')