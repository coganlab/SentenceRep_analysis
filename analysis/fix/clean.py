# %%
import os
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.mt_filter import line_filter
from analysis.fix.events import fix

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 57

# %% Load Data
layout = get_data("SentenceRep", LAB_root)
subj = f"D{subject:04}"
raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                      preload=True)

# %% filter data
line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
            filter_length='700ms', freqs=[60], notch_widths=20)
line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
            filter_length='20s', freqs=[60, 120, 180, 240],
            notch_widths=20)

# %% fix events
try:
    fixed = fix(raw)
    fixed.drop_channels('Trigger')
    del raw
except Exception as e:
    print(f"Error in {subj}: {e}")
    fixed = raw

# %% Save the data
save_derivative(fixed, layout, "clean", True)
