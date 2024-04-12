import os
from analysis.grouping import GroupData
from ieeg.viz.mri import plot_on_average
import numpy as np

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
base = GroupData.from_intermediates("SentenceRep", fpath, folder='stats',
                                    conds=dict(start=(-0.5, 0.5)))

cond = np.any(base.signif[0].astype(bool), axis=1)
chans = cond[cond].labels[0]
base.sig_chans = set(np.where(chans)[0])
plot_on_average(base.subjects, fpath + '/../ECoG_Recon', False, chans, hemi='both', color='yellow')

base.AUD = sub.AUD & base.sig_chans
base.SM = sub.SM & base.sig_chans
base.PROD = sub.PROD & base.sig_chans

print(f"Visual channels: {len(base.sig_chans)}/{len(cond)}")
print(f"Auditory/Visual channels: {len(base.AUD)}/{len(sub.AUD)}")
print(f"Sensory-Motor/Visual channels: {len(base.SM)}/{len(sub.SM)}")
print(f"Production/Visual channels: {len(base.PROD)}/{len(sub.PROD)}")