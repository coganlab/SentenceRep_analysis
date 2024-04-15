##
import os
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.mt_filter import line_filter
from ieeg.viz import mri
from events import fix_annotations, add_stim_conds
import matplotlib.pyplot as plt

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 19

## Load Data
layout = get_data("SentenceRep", LAB_root)
subjlist = layout.get_subjects()
subjlist.sort()
# subj = subjlist[subject]
fig, axs = plt.subplots(6, 6)
for subj, ax in zip(subjlist, axs.flat):
    # if int(subj[1:]) in (3, 32, 65, 71):
    #     ax.title.set_text(subj)
    #     continue
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                              extension='.edf', desc='clean', preload=False)
    # %%
    x = []
    for i, event in enumerate(filt.annotations):
        j = 0
        gonext = False
        if 'Response' in event['description'] and i != len(filt.annotations) - 1 and \
                not filt.annotations[i + 1]['description'].startswith('bad'):
            while not filt.annotations[i + j]['description'].startswith('Start'):
                j += 1
                if i + j == len(filt.annotations) - 1:
                    gonext = True
                    break
            if gonext:
                continue
            prev = filt.annotations[i + j]
            prev_off = prev['onset']
            ev_off = event['onset'] + event['duration']
            diff = prev_off - ev_off
            x.append(prev_off - ev_off)
    ax.hist(x, bins=20,
            range=(-0.5, 3))
    ax.title.set_text(subj)

