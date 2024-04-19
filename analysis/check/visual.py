import os
from analysis.grouping import GroupData
from ieeg.viz.mri import plot_on_average
import numpy as np
from analysis.utils.plotting import plot_channels

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
base = GroupData.from_intermediates("SentenceRep", fpath, folder='stats',
                                    conds=dict(start=(-0.5, 0.5)))

cond = np.any(base.signif[0].astype(bool), axis=1)
chans = cond[cond].labels[0]
base.sig_chans = set(np.where(cond)[0])
plot_on_average(base.subjects, fpath + '/../ECoG_Recon', False, chans, hemi='both', color='yellow')

base.AUD = sub.AUD & base.sig_chans
base.SM = sub.SM & base.sig_chans
base.PROD = sub.PROD & base.sig_chans

base.plot_groups_on_average(subj_dir=fpath + '/../ECoG_Recon', rm_wm=False, hemi='both')

# base.AUD = sub.AUD - base.sig_chans
# base.SM = sub.SM - base.sig_chans
# base.PROD = sub.PROD - base.sig_chans
#
# base.plot_groups_on_average(subj_dir=fpath + '/../ECoG_Recon', rm_wm=False, hemi='both')

print(f"Visual channels: {len(base.sig_chans)}/{len(cond)}")
print(f"Auditory/Visual channels: {len(base.AUD)}/{len(sub.AUD)}")
print(f"Sensory-Motor/Visual channels: {len(base.SM)}/{len(sub.SM)}")
print(f"Production/Visual channels: {len(base.PROD)}/{len(sub.PROD)}")

idx = sorted(base.SM)
plot_channels(sub.array["zscore","aud_ls"].combine((0,2))[idx,], sub.signif["aud_ls", idx], 6, 6, times=[-0.5, 1.5])
# plot_channels(base.array["zscore","start"].combine((0,2))[sorted(base.SM),], base.signif["start", sorted(base.SM)], 6, 6, times=[-0.5, 0.5])
