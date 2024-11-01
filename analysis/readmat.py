#
import os
import scipy.io
import numpy as np
from ieeg.io import get_data, raw_from_layout

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("Phoneme_sequencing", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

#%%
matdir = os.path.join(LAB_root, 'D_Data', 'Phoneme_Sequencing')
for subj in subjects:
    if (subj[1:] != 29):
        continue
    else:
        matpath = os.path.join(matdir, subj, )