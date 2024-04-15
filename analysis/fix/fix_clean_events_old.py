##
import os
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.mt_filter import line_filter
from ieeg.viz import mri
from events import fix

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 6

## Load Data
layout = get_data("SentenceRep", LAB_root)
subjlist = layout.get_subjects()
subjlist.sort()
# subj = subjlist[subject]
for subj in subjlist:
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                          preload=True)
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                              extension='.edf', desc='clean', preload=False)
    fixed = fix(raw)
    fixed.annotations._orig_time = filt.annotations.orig_time
    filt.set_annotations(fixed.annotations)
    # %% Save the data
    save_derivative(filt, layout, "clean", overwrite=True)