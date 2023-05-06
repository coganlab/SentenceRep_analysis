from ieeg.viz.utils import chan_grid
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data
from ieeg.calc.scaling import rescale
import os
from mne.time_frequency import tfr_morlet
import numpy as np

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 28

# Load the data
TASK = "SentenceRep"
subj = "D" + str(subject).zfill(4)
layout = get_data("SentenceRep", root=LAB_root)
filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                       extension='.edf', desc='clean', preload=False)

# %% fix SentenceRep events
from events import fix_annotations  # noqa E402
fix_annotations(filt)

# %% Crop raw data to minimize processing time
new = crop_empty_data(filt)

good = new.copy()

good.info['bads'] = channel_outlier_marker(new, 4)

good.drop_channels(good.info['bads'])
good.load_data()

ch_type = filt.get_channel_types(only_data_chs=True)[0]
good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

# Remove intermediates from mem
del new
# good.plot()

# %% High Gamma Filter and epoching
resp = trial_ieeg(good, "Word/Response", (-1.5, 1.5), preload=True, outliers=10)
base = trial_ieeg(good, "Start", (-1, 0.5), preload=True, outliers=10)

# %% create spectrograms
freqs = np.logspace(np.log10(2), np.log10(500), 100)
#
resp_s = tfr_morlet(resp, freqs, n_jobs=6, verbose=10, average=True,
                        n_cycles=freqs/2, return_itc=False,
                        decim=20)
resp_s.crop(tmin=-1, tmax=1)
base_s = tfr_morlet(base, freqs, n_jobs=6, verbose=10, average=True,
                        n_cycles=freqs/2, return_itc=False,
                        decim=20)
base_s.crop(tmin=-0.5, tmax=0)

spec = resp_s.copy()
spec._data = rescale(resp_s._data, base_s._data, mode='ratio', axis=(1, 2),
                     copy=True)

# %% plotting

figs = chan_grid(spec, vmin=0.7, vmax=1.4)
for i, f in enumerate(figs):
    f.savefig(os.path.join(layout.root, 'derivatives', 'figs', 'wavelet',
                           f'{subj}_response_{i}'))
