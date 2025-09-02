# %% Import dir
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import pickle

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
stim_path = os.path.join(LAB_root, 'BIDS-1.4_Phoneme_sequencing', 'BIDS', 'stimuli')
analysisfolder = 'SentenceRep_analysis\\analysis'

stim_list = ['abae','abi', 'aka', 'aku', 'ava', 'avae',
             'aeba', 'aebi', 'aebu', 'aega', 'aeka', 'aepi',
             'ibu', 'ika', 'ikae', 'ipu', 'iva', 'ivu',
             'uba', 'uga', 'ugae', 'ukae', 'upi', 'upu', 'uvae', 'uvi',
             'bab', 'baek', 'bak', 'bup',
             'gab', 'gaeb', 'gaev', 'gak', 'gav', 'gig', 'gip', 'gub',
             'kab', 'kaeg', 'kub', 'kug',
             'paek', 'paep', 'paev', 'puk', 'pup',
             'vaeg', 'vaek', 'vip', 'vug', 'vuk']

true_cat_vcv = {'abae':1, 'abi':1, 'aka':1, 'aku':1, 'ava':1, 'avae':1,
                 'aeba':1, 'aebi':1, 'aebu':1, 'aega':1, 'aeka':1, 'aepi':1,
                 'ibu':1, 'ika':1, 'ikae':1, 'ipu':1, 'iva':1, 'ivu':1,
                 'uba':1, 'uga':1, 'ugae':1, 'ukae':1, 'upi':1, 'upu':1, 'uvae':1, 'uvi':1,
                 'bab':2, 'baek':2, 'bak':2, 'bup':2,
                 'gab':2, 'gaeb':2, 'gaev':2, 'gak':2, 'gav':2, 'gig':2, 'gip':2, 'gub':2,
                 'kab':2, 'kaeg':2, 'kub':2, 'kug':2,
                 'paek':2, 'paep':2, 'paev':2, 'puk':2, 'pup':2,
                 'vaeg':2, 'vaek':2, 'vip':2, 'vug':2, 'vuk':2}

#%%
# Compute spectrogram using STFT
# S = librosa.stft(abae)
# S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
#
# # Display the spectrogram
# fig, axes = plt.subplots(2,1,figsize=(10, 4))
# librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma', ax=axes[0])
# axes[1].plot(np.abs(np.mean(S, axis =0)))
# plt.tight_layout()
# plt.show()

#%%
def extract_envelope(waveform, sampling_rate=16000, resample_rate=None):
    from scipy.signal import hilbert
    analytic_signal = hilbert(waveform)
    amplitude_envelope = np.abs(analytic_signal)
    # Resample if needed
    if resample_rate is not None:
        # Use librosa for resampling
        amplitude_envelope = librosa.resample(
            amplitude_envelope,
            orig_sr=sampling_rate,
            target_sr=resample_rate
        )
    return amplitude_envelope

n_stims = len(stim_list)
n_cols = 6  # number of columns
n_rows = int(np.ceil(n_stims / n_cols))
n_smooth = 5
n_timepoints = 150
padded_envelopes = {}

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 2))
axes = axes.flatten()  # so we can index directly

for i, stim in enumerate(stim_list):
    audio, sr = librosa.load(os.path.join(stim_path, f'{stim}.wav'))
    envelope = extract_envelope(audio, sr, resample_rate=100)
    smoothed = np.convolve(envelope, np.ones(n_smooth) / n_smooth, mode='same')

    mean_val = np.mean(smoothed)
    pad_end = 100-len(smoothed)
    padded_end = np.pad(smoothed, (0, pad_end), constant_values=0)
    padded = np.pad(padded_end, (50,0), constant_values=0)
    padded_envelopes[stim] = padded

    ax = axes[i]
    ax.plot(padded, linewidth=4)
    ax.set_title(stim, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(analysisfolder, 'stim_audio_envelopes.png'), dpi=300,
                bbox_inches='tight')
plt.show()

with open(f'{analysisfolder}\\padded_envelopes.pkl', 'wb') as f:
    pickle.dump(padded_envelopes, f)

#%%
with open(f'{analysisfolder}\\padded_envelopes.pkl', 'rb') as f:
    padded_envelopes = pickle.load(f)

binary_envelopes = {'vcv': np.zeros((150,)), 'cvc': np.zeros((150,))}
vcv_count = 0
cvc_count = 0

for item in padded_envelopes.keys():
    if true_cat_vcv[item] == 1:
        binary_envelopes['vcv'] += padded_envelopes[item]
        vcv_count += 1
    if true_cat_vcv[item] == 2:
        binary_envelopes['cvc'] += padded_envelopes[item]
        cvc_count += 1

binary_envelopes['vcv'] = binary_envelopes['vcv']/vcv_count
binary_envelopes['cvc'] = binary_envelopes['cvc']/cvc_count

with open(f'{analysisfolder}\\binary_envelopes.pkl', 'wb') as f:
    pickle.dump(binary_envelopes, f)


#%%
with open(f'{analysisfolder}\\padded_envelopes.pkl', 'rb') as f:
    padded_envelopes = pickle.load(f)

import matplotlib.pyplot as plt

# X-axis (151 points from -0.5 to 1)
x = np.linspace(-0.5, 1, 151)

# Pad all traces for plotting
vcv_traces = [np.append(padded_envelopes[item],0)
              for item in padded_envelopes.keys() if true_cat_vcv[item] == 1]
cvc_traces = [np.append(padded_envelopes[item],0)
              for item in padded_envelopes.keys() if true_cat_vcv[item] == 2]

# Mean traces (pad again to match x-axis)
vcv_mean = np.mean(vcv_traces,axis=0)
cvc_mean = np.mean(cvc_traces,axis=0)

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 300
})

# VCV subplot
for trace in vcv_traces:
    axes[0].plot(x, trace, color="steelblue", linewidth=0.5, alpha=0.3)
axes[0].plot(x, vcv_mean, color="navy", linewidth=1.0)
axes[0].set_title("VCV", fontsize=14)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude (a.u.)")
axes[0].set_xticks([-0.5,0,0.5,1])
axes[0].set_yticks([0,0.5])
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# CVC subplot
for trace in cvc_traces:
    axes[1].plot(x, trace, color="steelblue", linewidth=0.5, alpha=0.3)
axes[1].plot(x, cvc_mean, color="navy", linewidth=1.0)
axes[1].set_title("CVC", fontsize=14)
axes[1].set_xlabel("Time (s)")
axes[1].set_xticks([-0.5,0,0.5,1])
axes[1].set_yticks([0,0.5])
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# Tight layout
plt.tight_layout()

# Save as SVG
plt.savefig(f"{analysisfolder}\\binary_envelopes.svg", format="svg", dpi=300)
plt.close()
