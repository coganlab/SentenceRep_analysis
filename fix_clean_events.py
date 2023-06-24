##
import os
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.mt_filter import line_filter
from ieeg.viz import mri
from events import fix_annotations, add_stim_conds

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
subjlist.reverse()
# subj = subjlist[subject]
for subj in subjlist:
    if subj == "D0029":
        break
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                          preload=True)
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                              extension='.edf', desc='clean', preload=False)

    fix_annotations(raw)
    add_stim_conds(raw)
    filt.set_annotations(raw.annotations)

    # %% Save the data
    save_derivative(filt, layout, "events")