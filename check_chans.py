from ieeg.viz.utils import chan_grid
from ieeg.viz.parula import parula_map
from ieeg.io import get_data, raw_from_layout, update
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data
from ieeg.calc.scaling import rescale
from ieeg.calc.stats import avg_no_outlier, find_outliers
import os
from mne.time_frequency import tfr_morlet
from ieeg.timefreq.utils import wavelet_scaleogram
import numpy as np

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 3

# Load the data
TASK = "SentenceRep"
subj = "D" + str(subject).zfill(4)
layout = get_data("SentenceRep", root=LAB_root)
filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                       extension='.edf', desc='clean', preload=False)

# %% fix SentenceRep events
from events import fix_annotations  # noqa E402
new = crop_empty_data(filt,)

good = new.copy()
fix_annotations(good)

# %% Crop raw data to minimize processing time

# good.drop_channels(good.info['bads'])
# good.info['bads'] += channel_outlier_marker(good, 3, 2, save=True)
good.drop_channels(good.info['bads'])
# good.info['bads'] += channel_outlier_marker(good, 4, 2)
# good.drop_channels(good.info['bads'])
good.load_data()

ch_type = filt.get_channel_types(only_data_chs=True)[0]
good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

# Remove intermediates from mem
del new
# good.plot()

# %% epoching and trial outlier removal

resp = trial_ieeg(good, "Response", (-1.5, 1.5), preload=True)
resp_gtrials = find_outliers(resp.get_data(), 11)
resp_func = lambda x: avg_no_outlier(x, keep=resp_gtrials)
# resp_avg = resp.average(None, resp_func)

base = trial_ieeg(good, "Start", (-1, 0.5), preload=True)
base_gtrials = find_outliers(base.get_data(), 8)
base_func = lambda x: avg_no_outlier(x, keep=base_gtrials)
# base_avg = base.average(None, base_func)
# scaled = rescale(resp, base, mode='zscore', copy=True, outliers=10)

# %% create spectrograms

resp_s = wavelet_scaleogram(resp, n_jobs=-2, decim=int(good.info['sfreq']/100))
resp_s.crop(tmin=-1, tmax=1)
base_s = wavelet_scaleogram(base, n_jobs=-2, decim=int(good.info['sfreq']/100))
base_s.crop(tmin=-0.5, tmax=0)

spec = rescale(resp_s, base_s, copy=True, mode='ratio',
               base_trials=base_gtrials)
spec_a = spec.average(resp_func, copy=True)
spec_a._data = np.log10(spec_a._data) * 20

# %% plotting

figs = chan_grid(spec_a, size=(16, 12), vmin=-2, vmax=2,
                 cmap=parula_map)
for i, f in enumerate(figs):
    f.savefig(os.path.join(layout.root, 'derivatives', 'figs', 'wavelet',
                           f'{subj}_response_{i + 1}'))
assert False

# %% save bad channels
good.info['bads'] += spec.info['bads']
update(good, "muscle")
