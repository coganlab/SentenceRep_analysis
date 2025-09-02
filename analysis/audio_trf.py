import os
import numpy as np
from isort.core import process
from matplotlib import pyplot as plt
import pickle
import mtrf
from ieeg.io import get_data

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    RECON_root = os.path.join(HOME, "Box", "ECoG_Recon")
layout = get_data("Phoneme_sequencing", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")
DData_dir = os.path.join(LAB_root, 'D_Data', 'Phoneme_Sequencing')
analysisfolder = 'SentenceRep_analysis\\analysis'

with open(f'{analysisfolder}\\padded_envelopes.pkl', 'rb') as f:
    padded_envelopes = pickle.load(f)

#%% Prep data to labeledarray
from ieeg.calc.mat import LabeledArray, combine
from analysis.decoding import Decoder
from analysis.utils.mat_load import load_dict
from analysis.check.chan_utils import nested_dict_to_ndarray

suffix = "zscore-epo.fif"
conds = {'aud': (-0.5,1), 'go': (-0.5,1), 'resp': (-0.5,1)}

zscores = load_dict(layout, conds, 'zscore', False, 'stgrerefgammastats')
mask = load_dict(layout, conds, 'significance', True, 'stgrerefgammastats')
zscores = combine(zscores, (0, 3)) # combine subj with channel
zscoresArray, zscoresLabel = nested_dict_to_ndarray(zscores)
zscoresLA = LabeledArray(zscoresArray, labels=zscoresLabel)
mask = combine(mask, (0, 2)) # combine subj with channel as stims already averaged
# maskArray, maskLabel = nested_dict_to_ndarray(mask)
# maskLA = LabeledArray(maskArray, labels=maskLabel)
maskLA = LabeledArray.from_dict(mask)
sig_idx = dict()
# get sig chans for each cond
for i, cond in enumerate(conds.keys()):
    if cond not in sig_idx:
        sig_idx[cond] = {}
    sig_idx[cond] = np.where(np.any(maskLA.__array__()[i,:,:] == 1, axis = 1))[0].tolist()
# get union of sig chans
sig_idx_union = list(set(sig_idx['aud'])|set(sig_idx['go'])|set(sig_idx['resp']))
zscoresLA = zscoresLA.combine((1,3))

#%% checking fit, not used
i_cond = 0
fs = 100
tmin, tmax = -0.1, 0.7
zscoresLA_cond = zscoresLA.take(i_cond, axis=0)
n_channels = zscoresLA_cond.shape[0]
channels_per_fig = 50
regularization = np.logspace(-4, 5, 10)
n_folds = 5
plt.figure(figsize=(10, 5))
all_weights = []
inspected_channels = [18, 19, 20, 21, 22, 53, 54, 57, 60, 89, 90, 91, 94, 95, 113]

for ch_idx in range(n_channels):
    zscoresLA_cond_ch = zscoresLA_cond.take(ch_idx, axis=0)
    valid_epoch_idx = np.array(~np.any(np.isnan(zscoresLA_cond_ch.__array__()), axis=1))
    valid_epoch_names = zscoresLA_cond_ch.labels[0][valid_epoch_idx]
    stimulus = [padded_envelopes[item.split('-')[0]] for item in valid_epoch_names]
    response = zscoresLA_cond_ch.__array__()[valid_epoch_idx]
    response = [row[:, np.newaxis] for row in response]

    # r_fwd = mtrf.stats.crossval(ch_trf, stimulus, response, fs, tmin, tmax, reg_lambda, k=10)
    # print(f"mean correlation between actual and predicted response: {r_fwd.mean().round(3)}")

    ch_trf = mtrf.model.TRF(direction=1)
    ch_trf.train(stimulus, response, fs, tmin, tmax, regularization)
    all_weights.append(ch_trf.weights.squeeze())
    if ch_idx in inspected_channels:
        plt.plot(ch_trf.times, ch_trf.weights.squeeze(), alpha = 0.3, color='green')
    else:
        plt.plot(ch_trf.times, ch_trf.weights.squeeze(), alpha=0.3, color='blue')
    # Plot weights for the first feature (e.g., single audio envelope)

mean_weights = np.mean(np.stack(all_weights), axis=0)
plt.plot(ch_trf.times, mean_weights, color='black', linewidth=2, label='Mean TRF')

plt.title("Superimposed TRF Weights Across Channels")
plt.xlabel("Time lag (s)")
plt.ylabel("TRF Weight")
plt.grid(True)
plt.tight_layout()
plt.show()


#%% some PCA prep
#Channel stim level setup
from analysis.decoding import classes_from_labels
from analysis.check.chan_utils import remove_min_nan_ch, equal_valid_trials_ch, left_adjust_by_stim
true_cat_vcv = {'abae':1, 'abi':1, 'aka':1, 'aku':1, 'ava':1, 'avae':1,
                 'aeba':1, 'aebi':1, 'aebu':1, 'aega':1, 'aeka':1, 'aepi':1,
                 'ibu':1, 'ika':1, 'ikae':1, 'ipu':1, 'iva':1, 'ivu':1,
                 'uba':1, 'uga':1, 'ugae':1, 'ukae':1, 'upi':1, 'upu':1, 'uvae':1, 'uvi':1,
                 'bab':2, 'baek':2, 'bak':2, 'bup':2,
                 'gab':2, 'gaeb':2, 'gaev':2, 'gak':2, 'gav':2, 'gig':2, 'gip':2, 'gub':2,
                 'kab':2, 'kaeg':2, 'kub':2, 'kug':2,
                 'paek':2, 'paep':2, 'paev':2, 'puk':2, 'pup':2,
                 'vaeg':2, 'vaek':2, 'vip':2, 'vug':2, 'vuk':2}
with open(f'{analysisfolder}\\binary_envelopes.pkl', 'rb') as f:
    binary_envelopes = pickle.load(f)

def batch_concatenate(data, group_size=5, drop_incomplete=True):
    """
    Concatenate every `group_size` consecutive 1D arrays in `data` into one long array.
    """
    result = []
    for i in range(0, len(data), group_size):
        chunk = data[i:i + group_size]
        if drop_incomplete and len(chunk) < group_size:
            continue
        result.append(np.concatenate(chunk))
    return result

iter_num = 1 # subsampling
trial_num = 50
concat_trial_num = 5
encoder_cat = {'vcv': 1, 'cvc': 2}
encoder_cat_flipped = {v: k for k, v in encoder_cat.items()}
weights_out = {}

#TRF setup
i_cond = 2
fs = 100
tmin, tmax = -0.1, 0.7
zscoresLA_cond = zscoresLA.take(i_cond, axis=0)
regularization = np.logspace(-4, 5, 10)
n_folds = 5

cats, labels = classes_from_labels(zscoresLA_cond.labels[1], crop=slice(0, 4)) #this get out repetitions of same stims
flipped_cats = {v:k for k,v in cats.items()} #{0: abae, 1: abi, 2: aeba, 3: aebi, 4: aebu, 5: aega, ...}
new_labels = np.array([true_cat_vcv[flipped_cats[l]] for l in labels]) #convert to true categories
zscoresLA_cond_idx, _ = remove_min_nan_ch(zscoresLA_cond, new_labels, min_non_nan=trial_num)
for i_iter in range(iter_num):
    zscoresLA_cond_reduced = equal_valid_trials_ch(zscoresLA_cond_idx, new_labels, min_non_nan=trial_num, upper_limit=trial_num)
    zscores_cropped, labels_cropped = left_adjust_by_stim(zscoresLA_cond_reduced, new_labels, crop=True)
    valid_labels = set(encoder_cat.values())
    valid_idx = np.where(labels_cropped[np.isin(labels_cropped, list(valid_labels))])[0]
    labels_cropped = labels_cropped[valid_idx]
    zscores_cropped = zscores_cropped.take(valid_idx, axis=1)
#------------
    stimulus = [binary_envelopes[encoder_cat_flipped[item]] for item in labels_cropped]
    response = zscores_cropped.__array__().transpose(1,2,0)
    response = [row for row in response]
    stimulus_chunked = batch_concatenate(stimulus, group_size = concat_trial_num, drop_incomplete=True)
    response_chunked = batch_concatenate(response, group_size = concat_trial_num, drop_incomplete=True)
    # r_fwd = mtrf.stats.crossval(ch_trf, stimulus, response, fs, tmin, tmax, reg_lambda, k=10)
    # print(f"mean correlation between actual and predicted response: {r_fwd.mean().round(3)}")

    roi_trf = mtrf.model.TRF(direction=1)
    roi_trf.train(stimulus_chunked, response_chunked, fs, tmin, tmax, regularization, k=n_folds)
    roi_trf.plot()
    # all_weights.append(ch_trf.weights.squeeze())
    # if ch_idx in inspected_channels:
    #     plt.plot(ch_trf.times, ch_trf.weights.squeeze(), alpha = 0.3, color='green')
    # else:
    #     plt.plot(ch_trf.times, ch_trf.weights.squeeze(), alpha=0.3, color='blue')
    # Plot weights for the first feature (e.g., single audio envelope)
# mean_weights = np.mean(np.stack(all_weights), axis=0)
# plt.plot(ch_trf.times, mean_weights, color='black', linewidth=2, label='Mean TRF')
#
# plt.title("Superimposed TRF Weights Across Channels")
# plt.xlabel("Time lag (s)")
# plt.ylabel("TRF Weight")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#%%
from scipy.stats import bootstrap

def compute_mean_sem(data, axis=0):
    """Return mean and standard error of the mean along specified axis."""
    mean = np.mean(data, axis=axis)
    sem = np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])
    return mean, sem

def compute_bootstrap_sem(data1, data2, n_resamples=1000):
    """Bootstrap SEM for the difference of means between two sets (over axis=0)."""
    # Difference across samples (shape: samples x time)
    diff = data1 - data2

    # Reshape to (n_timepoints, n_samples) for bootstrap
    diff = np.swapaxes(diff, 0, 1)

    sems = []
    for t in diff:
        res = bootstrap((t,), np.mean, confidence_level=0.95, n_resamples=n_resamples, method='basic')
        sems.append((res.standard_error))

    return np.mean(diff, axis=1), np.array(sems)

def plot_with_sem(ax, x, mean, sem, label=None, color=None):
    """Plot mean ± SEM on a given Axes."""
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - sem, mean + sem, alpha=0.3, color=color)



def process_and_plot(data):
    """
    Given a 3D array of shape (subjects, 100, 150), generate 3 subplots:
    - Top 50 trials
    - Bottom 50 trials
    - Difference (Top - Bottom) with bootstrapped SEM
    """
    x = np.linspace(-0.5, 1.0, data.shape[2])
    n_ch = data.shape[0]

    # Set up figure with improved styling
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    titles = ["VCV", "CVC", "VCV-CVC"]
    xticks = [-0.5, 0, 0.5, 1.0]

    for ax in axs:
        ax.set_xticks(xticks)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)

    # Plot per-subject data
    for i in range(n_ch):
        ch_data = data[i]  # shape: (100, 150)
        top = ch_data[:50]
        bottom = ch_data[50:]

        # Compute means and SEMs
        mean_top, sem_top = compute_mean_sem(top, axis=0)
        mean_bot, sem_bot = compute_mean_sem(bottom, axis=0)
        mean_diff, sem_diff = compute_bootstrap_sem(top, bottom)

        # Plot with original style
        plot_with_sem(axs[0], x, mean_top, sem_top)  # VCV
        plot_with_sem(axs[1], x, mean_bot, sem_bot)  # CVC
        plot_with_sem(axs[2], x, mean_diff, sem_diff)  # Diff

    # Titles and labels
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=11)
    axs[0].set_ylabel("High Gamma (Z-score)", fontsize=11)
    plt.savefig(f"{analysisfolder}\\pca_stgrerefgamma.svg", format="svg", dpi=300)
    plt.show()

process_and_plot(zscores_cropped.__array__())

#%%
# Define function to apply PCA to each run's data
from sklearn.decomposition import PCA, NMF
# # Apply PCA to EEG data
# zscores_cropped_pca = apply_pca_flexible(zscores_cropped.__array__(), axis = 0, n_components = 10)
from matplotlib.colors import to_rgb
import colorsys
def generate_shaded_colors(base_color, n):
    """Generate 'n' perceptually distinct variations of a base color"""
    base_rgb = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    lightness_vals = np.linspace(0.25, 0.85, n)  # wider range for stronger contrast
    colors = [colorsys.hls_to_rgb(h, l_val, s) for l_val in lightness_vals]
    return colors

def apply_pca_with_channel_loadings(data, label_names, axis=0, n_components=10, show_pies=True, pie_threshold=0.01):
    """
    Apply PCA to a 3D array (n_channels, n_trials, n_samples) along axis 0 (channels) or 1 (trials),
    return transformed data plus normalized squared loadings for the decomposed axis, and optionally
    plot pie charts for the first components.

    Parameters:
        data (np.ndarray): Input array with shape (n_channels, n_trials, n_samples)
        label_names (list of str): Names corresponding to the axis being decomposed:
            - if axis=0: list of channel names of length n_channels
            - if axis=1: list of trial identifiers of length n_trials
        axis (int): 0 for PCA across channels, 1 for PCA across trials
        n_components (int): Number of principal components to retain
        show_pies (bool): Whether to display pie charts for the first up to 10 components
        pie_threshold (float): Minimum proportion to show a label in pie chart (default 1%)

    Returns:
        transformed_data (np.ndarray):
            - if axis=0: shape (n_components, n_trials, n_samples)
            - if axis=1: shape (n_channels, n_components, n_samples)
        pca (sklearn.decomposition.PCA): Fitted PCA object
        normalized_squared_loadings (np.ndarray): Array of shape (n_components, len(label_names))
            giving each original label's squared contribution to each component (rows sum to 1)
    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 (channels) or 1 (trials)")

    # Prepare feature matrix: samples x features
    if axis == 0:
        # PCA across channels: features = channels
        n_channels, n_trials, n_samples = data.shape
        X = data.reshape(n_channels, -1).T  # (n_trials * n_samples, n_channels)
    else:
        # PCA across trials: features = trials
        data_trans = np.transpose(data, (1, 0, 2))  # (n_trials, n_channels, n_samples)
        n_trials, n_channels, n_samples = data_trans.shape
        X = data_trans.reshape(n_trials, -1).T  # (n_channels * n_samples, n_trials)

    # Fit PCA

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)  # (n_observations, n_components)
    # nmf = NMF(n_components=n_components, init="nndsvda", random_state=0)
    # scores = nmf.fit_transform(X)

    # Extract loadings: components_ is (n_components, n_features)
    loadings = pca.components_  # signed weights
    abs_squared = np.square(loadings)
    normalized_squared_loadings = abs_squared / np.sum(abs_squared, axis=1, keepdims=True)  # rows sum to 1

    # Reformat transformed data to match expected shape
    if axis == 0:
        # Want (n_components, n_trials, n_samples)
        transformed_data = scores.T.reshape(n_components, n_trials, n_samples)
    else:
        # Want (n_channels, n_components, n_samples)
        temp = scores.T.reshape(n_components, n_channels, n_samples)  # (components, channels, samples)
        transformed_data = np.transpose(temp, (1, 0, 2))  # (channels, components, samples)

    explained_variance = 100 * pca.explained_variance_ratio_

    if show_pies:
        n_pies = min(n_components, 10)
        base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                       'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                       'tab:olive', 'tab:cyan']

        fig, axs = plt.subplots(n_pies, 1, figsize=(4, 2 * n_pies), constrained_layout=True)

        for comp in range(n_pies):
            ax = axs[comp] if n_pies > 1 else axs
            slices = normalized_squared_loadings[comp]
            display_labels = [lbl if val >= pie_threshold else '' for lbl, val in zip(label_names, slices)]
            pie_colors = generate_shaded_colors(base_colors[comp % len(base_colors)], len(slices))

            wedges, texts, autotexts = ax.pie(
                slices,
                labels=display_labels,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else '',
                colors=pie_colors,
                startangle=90,
                textprops={'fontsize': 6},  # smaller font for labels
                wedgeprops=dict(linewidth=0)  # no white borders
            )
            ax.set_title(f"PC {comp + 1} ({explained_variance[comp]:.2f}% total variance)", fontsize=8)

        plt.suptitle("Electrode Loadings in PC", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{analysisfolder}\\pc_loadings_stgrerefgamma_resp.svg", format="svg", dpi=300)
        plt.show()

    return transformed_data, pca, normalized_squared_loadings

ch_name_list = zscoresLA_cond_reduced.labels[0]
#indices = [i for i, s in enumerate(ch_name_list) if 'D0042-RPST' not in s]
pca_data, pca_fitted, _ = apply_pca_with_channel_loadings(zscores_cropped.__array__(), ch_name_list, n_components=4)
process_and_plot(pca_data)

#%%
loadings = pca_fitted.components_
notable_ch_idx = set().union(*[
    np.where(loadings[i, :] > 0.01)[0] for i in range(4)
])
comp_idx = {0:[], 1:[], 2:[], 3:[]}
for i in notable_ch_idx:
    max_comp_loading = np.argmax(loadings[:4,i])
    comp_idx[max_comp_loading].append(i)
comp_names = {i: ch_name_list[comp_idx[i]] for i in range(4)}
comp_idxs = {
    comp: [i for i, name in enumerate(sub_channels) if name in comp_names[comp]]
    for comp in comp_names
}

#%% CV for regularization and inspection
# Determine number of figures needed
n_figs = (n_channels + channels_per_fig - 1) // channels_per_fig  # equivalent to ceil(n_channels / channels_per_fig)

for fig_idx in range(n_figs):
    start_ch = fig_idx * channels_per_fig
    end_ch = min(start_ch + channels_per_fig, n_channels)
    n_subplots = end_ch - start_ch

    n_cols = 5
    n_rows = -(-n_subplots // n_cols)  # integer ceiling without math.ceil

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, ch in enumerate(range(start_ch, end_ch)):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        zscoresLA_cond_ch = zscoresLA_cond.take(ch, axis=0)
        valid_epoch_idx = np.array(~np.any(np.isnan(zscoresLA_cond_ch.__array__()), axis=1))
        valid_epoch_names = zscoresLA_cond_ch.labels[0][valid_epoch_idx]
        stimulus = [padded_envelopes[item.split('-')[0]] for item in valid_epoch_names]
        response = zscoresLA_cond_ch.__array__()[valid_epoch_idx]
        response = [row[:, np.newaxis] for row in response]
        # Train with MSE (neg_mse → multiply by -1)
        trf_mse = mtrf.model.TRF(metric=mtrf.stats.neg_mse)
        mse = trf_mse.train(stimulus, response, fs, tmin, tmax, regularization, k=n_folds) * -1

        # Train with Pearson's r
        trf_r = mtrf.model.TRF(metric=mtrf.stats.pearsonr)
        r = trf_r.train(stimulus, response, fs, tmin, tmax, regularization, k=n_folds)

        # Plot both
        ax1.semilogx(regularization, r, color='c', label='r')
        ax2.semilogx(regularization, mse, color='m', label='MSE')
        ax1.axvline(regularization[np.argmin(mse)], linestyle='--', color='k', linewidth=0.8)

        ax1.set_title(f'Channel {ch}', fontsize=9)
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'TRF Regularization Tuning (Channels {start_ch}–{end_ch - 1})', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



