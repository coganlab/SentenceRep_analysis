## Description: Check channels for outliers and remove them
import mne.time_frequency
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from ieeg.io import get_data, update, get_bad_chans
from ieeg.calc.scaling import rescale
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

## check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject =53


# Load the data
TASK = "SentenceRep"
layout = get_data("SentenceRep", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")
conds = ["start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_jl", "go_ls", "go_lm"]

for subj in subjects:
    if int(subj[1:]) != subject:
        continue
    for cond in conds:
        spec_type = 'wavelet_test'
        filename = os.path.join(layout.root, 'derivatives',
                                'spec', spec_type, subj, f'{cond}-tfr.h5')
        # filename = os.path.join(layout.root, 'derivatives', 'stats_freq',
        #                         f'{subj}_{cond}_{spec_type}-tfr.h5')
        try:
            spec = mne.time_frequency.read_tfrs(filename)
            if isinstance(spec, mne.time_frequency.tfr.EpochsTFR):
                spec_a = spec.average(lambda x: np.nanmean(x, axis=0))
            else:
                spec_a = spec
        except OSError:
            print(f"Skipping {subj} {cond}")
            continue
        if cond == 'start':
            base = spec_a.copy().crop(tmin=-0.5, tmax=0)

        # info_file = os.path.join(layout.root, spec_a.info['subject_info']['files'][0])
        # all_bad = get_bad_chans(info_file)
        # spec_a.info.update(bads=[b for b in all_bad if b in spec_a.ch_names])
        # rescale(spec_a, base, mode='ratio', copy=False)
        # spec_a._data = np.where(spec_a._data > 0.9999, 0.9999, spec_a._data)
        # spec_a._data = st.norm.ppf(1 - spec_a._data)

        ## plotting
        figs = chan_grid(spec_a, size=(20, 10),
                         vlim=(0., 1.),
                         cmap=parula_map, show=False)
        fig_path = os.path.join(layout.root, 'derivatives', 'spec', spec_type, 'figs')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        for i, f in enumerate(figs):
            f.savefig(os.path.join(fig_path, f'{subj}_{cond}_{i + 1}.jpg'), bbox_inches='tight')
            plt.close(f)

        ## save bad channels
        # update(spec, layout, "muscle")
