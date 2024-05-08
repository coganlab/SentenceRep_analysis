## Description: Check channels for outliers and remove them
import mne.time_frequency
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from ieeg.io import get_data, update, get_bad_chans
import os
import matplotlib.pyplot as plt

## check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 5


# Load the data
TASK = "SentenceRep"
layout = get_data("SentenceRep", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")
conds = ["start", "aud_ls", "aud_lm", "aud_jl", "go_jl", "go_ls", "go_lm", "resp"]

for subj in subjects:
    for cond in conds:
        spec_type = 'multitaper_test'
        filename = os.path.join(layout.root, 'derivatives',
                                'spec', spec_type, subj, f'{cond}-tfr.h5')
        try:
            spec = mne.time_frequency.read_tfrs(filename)
        except FileNotFoundError:
            print(f"Skipping {subj} {cond}")
            continue
        info_file = os.path.join(layout.root, spec.info['subject_info']['files'][0])
        all_bad = get_bad_chans(info_file)
        spec.info.update(bads=[b for b in all_bad if b in spec.ch_names])

        ## plotting
        import matplotlib as mpl
        figs = chan_grid(spec, size=(20, 10), vlim=(0.7, 1.4),
                         cmap=parula_map, show=False)
        fig_path = os.path.join(layout.root, 'derivatives', 'figs', spec_type)
        for i, f in enumerate(figs):
            f.savefig(os.path.join(fig_path, f'{subj}_{cond}_{i + 1}.jpg'), bbox_inches='tight')
            plt.close(f)

        ## save bad channels
        # update(spec, layout, "muscle")
