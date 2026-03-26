## Preprocess
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, outliers_to_nan, trial_ieeg
from ieeg.timefreq import gamma, utils
from ieeg.calc import stats
from ieeg.arrays import reshape
import os
from itertools import product
import numpy as np


## check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")

nchans = 0
nbads = 0
for subj in subjects:
    if int(subj[1:]) in (3, 30, 32, 65, 71):
        continue
    # Load the data
    TASK = "SentenceRep"
    # subj = "D" + str(sub).zfill(4)
    layout = get_data("SentenceRep", root=LAB_root)
    filt = raw_from_layout(layout.derivatives['notch'], subject=subj,
                           extension='.edf', desc='notch', preload=False)

    nchans += len(filt.ch_names)
    nbads += len(filt.info['bads'])