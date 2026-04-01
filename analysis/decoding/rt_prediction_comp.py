# Perform a linear regression decoding analysis to predict reaction times from neural data.

import numpy as np
import os

from analysis.grouping import group_elecs
import torch
import slicetca
from functools import partial
from ieeg.io import get_data
from analysis.load import load_data, load_spec, exclude
from analysis.utils import load_response_times
from ieeg.arrays.label import LabeledArray
from ieeg.viz.ensemble import plot_dist
from ieeg.calc.stats import time_perm_cluster, ridge_nd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from analysis.utils.plotting import plot_horizontal_bars
from scipy.interpolate import interp1d

def get_unique_subjects(subset, all_chans):
    subjects = {}
    for subset_ch_idx, ch_identifier in enumerate(subset):
        global_ch_idx = all_chans.index(ch_identifier)
        if global_ch_idx is None:
            print(f"Warning: Channel identifier '{ch_identifier}' not found in all_channel_labels, skipping")
            continue

        ch_name = str(all_chans[global_ch_idx])
        subj = ch_name.split('-')[0] if '-' in ch_name else ch_name
        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(subset_ch_idx)
    return subjects

def make_rt_array(rt_dict, trial_df_dict, labels, unique_subjects, exclude):
    # make sure rt_array matches in_data trials
    rt = LabeledArray(
        np.full([len(n) for n in labels], np.nan),
        labels
    )
    # Iterate over unique subjects
    for subj, subset_ch_indices in unique_subjects.items():
        if subj in exclude:
            continue

        # Check if any channel for this subject is excluded
        for subset_ch_idx in subset_ch_indices:
            ch_identifier = labels[0][subset_ch_idx]
            if ch_identifier in exclude:
                subset_ch_indices.pop(subset_ch_indices.index(subset_ch_idx))
                continue

        trial_df = trial_df_dict.get(subj)
        rt_array1 = rt_dict.get(subj)

        if rt_array1 is None or len(rt_array1) == 0 or trial_df is None:
            # Raise error if whole subject is missing RT data
            raise ValueError(
                f"Subject {subj}: Missing RT data for response-aligned weighting. "
                f"RT data is required for all subjects in response-aligned data.")

        # Group RT data by token and match sequentially within each token
        rt_by_token = {}
        for _, row in trial_df.iterrows():
            token_with_prefix = str(row['token']).strip()
            if '/' in token_with_prefix:
                token = token_with_prefix.split('/')[-1].lower()
            else:
                token = token_with_prefix.lower()
            rt_val = row['response_time']

            if token not in rt_by_token:
                rt_by_token[token] = []
            rt_by_token[token].append(rt_val)

        # Create mapping from (token, trial_index_in_neural_data) to RT
        rt_map = {}
        for token, rt_values in rt_by_token.items():
            for trial_idx, rt_val in enumerate(rt_values):
                rt_map["-".join([token, str(trial_idx)])] = rt_val

        print(f"Subject {subj}: Created RT map with {len(rt_map)} entries from"
              f" {len(rt_by_token)} tokens for resp data")

        for ch_n in subset_ch_indices:
            for t, rt_val in rt_map.items():
                ch = labels[0][ch_n]
                rt[ch, t] = rt_val
    return rt


def weighted_preserve_stats(data, weights, axis=None):
    """
    Multiplies data along the specified axis by weights, then rescales
    to preserve the original mean and variance.

    Parameters:
        data (np.ndarray): The input data array.
        weights (np.ndarray): The weight vector.
        axis (int): The axis along which to multiply.

    Returns:
        np.ndarray: The weighted and rescaled data.
    """
    # Check for valid weights
    weights = np.asarray(weights)
    if weights.size == 0:
        return

    # Check for all-zero or all-NaN weights
    weights_mean = np.nanmean(weights)
    if weights_mean == 0 or np.isnan(weights_mean):
        # If weights are invalid, don't modify data
        return

    # Normalize weights
    w = weights / weights_mean

    # Ensure weights match data dimension along specified axis
    if axis is not None:
        if axis < 0:
            axis = data.ndim + axis
        if data.shape[axis] != weights.size:
            raise ValueError(
                f"Weights size ({weights.size}) doesn't match data dimension "
                f"along axis {axis} ({data.shape[axis]})")

    # Multiply along the specified axis
    if axis is None:
        data *= w
    else:
        w_reshaped = w.reshape(
            [1 if i != axis else -1 for i in range(data.ndim)])
        data *= w_reshaped

HOME = os.path.expanduser("~")
print("HOME directory: {}", HOME)
if False:
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    # n = int(os.environ['SLURM_ARRAY_TASK_ID'])
    # print(n)
    n = 1
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    n = 1

log_dir = os.path.join(os.path.dirname(LAB_root), 'logs', str(n))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print("Log directory: {}", log_dir)
layout = get_data('SentenceRep', root=LAB_root)
print("Loaded layout: {}", getattr(layout, 'root', str(layout)))

conds_all = {"aud_ls": (-0.5, 1.5),
             "aud_lm": (-0.5, 1.5),
             "aud_jl": (-0.5, 1.5),
             "go_ls": (-0.5, 1.5),
             "go_lm": (-0.5, 1.5),
             "go_jl": (-0.5, 1.5),
             }

group = 'SM'
folder = 'stats_freq_hilbert'

sigs = load_data(layout, folder, "mask")
print("Loaded sigs: type={} labels={}", type(sigs), getattr(sigs, 'labels', None))
AUD, SM, PROD, sig_chans, delay = group_elecs(sigs,
                                              [s for s in
                                               sigs.labels[1]
                                               if s not in exclude],
                                              sigs.labels[0])
print("Grouped electrodes: AUD={} SM={} PROD={} sig_chans={} delay={}", len(AUD),
   len(SM), len(PROD), len(sig_chans), len(delay))
idxs = {'SM': sorted(SM), 'AUD': sorted(AUD), 'PROD': sorted(PROD),
        'sig_chans': sorted(sig_chans), 'delay': sorted(delay)}
# idx = idxs[group]
zscores = load_data(layout, folder, "zscore")
print("Loaded zscores: {}", type(zscores))

n_components = (4,)
best_seed = 123457
window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': 1,
                 'average_repetitions': False, 'step': 8}

# #
# %% decompose the optimal model

import pickle

filename = f"{group}_chns.pkl"
if os.path.exists(filename):
    labels = pickle.load(open(filename, 'rb'))
    print("Loaded labels from pickle: {} entries",
       len(labels[0]) if hasattr(labels, '__len__') else 'unknown')
else:
    labels = load_spec(group, list(conds_all.keys()), layout, folder=folder,
                       min_nan=1, n_jobs=-2)[-2]
    pickle.dump(labels, open(filename, 'wb'))
    print("Computed and saved labels to {}", filename)
state = torch.load('model_SM2_freq.pt')
print("Loaded torch state keys: {}", list(state.keys())[:10])
shape = state['vectors.0.0'].shape[1:] + state['vectors.0.1'].shape[1:]
n_components = (state['vectors.0.0'].shape[0],)
model = slicetca.core.SliceTCA(
    dimensions=shape,
    ranks=(n_components[0], 0, 0, 0),
    positive=True,
    initialization='uniform-positive',
    dtype=torch.float32,
    lr=5e-4,
    weight_decay=partial(torch.optim.Adam, eps=1e-9),
    loss=torch.nn.L1Loss(reduction='mean'),
    init_bias=0.1,
    threshold=None,
    patience=None
)
model.load_state_dict(state)
print("Model loaded into slicetca.SliceTCA; n_components={}", n_components)
colors = ['orange', 'k', 'c', 'y']
names = ['Auditory', 'WM', 'Motor', 'Visual']
W, H = model.get_components(numpy=True)[0]
print(f"Extracted components W shape={getattr(W, 'shape', None)} H shape={getattr(H, 'shape', None)}")

# Load response times for response-aligned weighting
subjects_list_rt = list(
    set(s.split('-')[0] for s in sigs.labels[1] if s not in exclude))
rt_dict, trial_df_dict = load_response_times(layout, subjects_list_rt)
print(f"Loaded RT data for {sum(1 for v in rt_dict.values() if v is not None)} subjects")


# %%
go_weight_timing = (-0.5, 1.25)
go_timing = (-0.5, 1.5)
resp_timing = (-1, 1)

for cond, timing in zip(('go_ls',
                         # 'resp'
                         ),
                        (go_timing,
                         # resp_timing
                         )):
    print(f"Processing condition: {cond} with timing {timing}")
    # sub-select data to analyze
    # component loop
    corrs = []
    betas = []
    intercepts = []
    sigs = []
    uv_betas_all = []
    uv_r2_all = []
    uv_sig_all = []
    uv_labels_all = []
    n_permutations = 1000
    threshold = None  # use default threshold
    for i in range(n_components[0]):
        subset = np.nonzero(W[i] / W.mean() > 0.05)[0]
        comp_chans = [labels[0][s] for s in subset]
        new_labels = [comp_chans, labels[2]]
        in_data_full = zscores[cond][:, comp_chans, ..., :175].combine((0, 3)).dropna()

        print("{ } component, {} channels", names[i], len(subset))

        weights = model.construct_single_component(0, i).detach().numpy()[
            subset]
        weights_go = weights[:, 0, :, None, 175:]
        # Map subset channel indices to subjects
        unique_subjects = get_unique_subjects(comp_chans, zscores.labels[2])

        # Get tokens from neural data
        # Data structure: (tokens, channels, freq, trials, time)
        data_tokens = ['heat', 'hoot', 'hot', 'hut']
        rt_array = make_rt_array(rt_dict, trial_df_dict, new_labels,
                                 unique_subjects, exclude)

        if cond == 'resp':
            # Build RT array using frequency-averaged data for shape/labels
            print(f"Number of trials from labels[1]: {len(in_data.labels[1])}")


            # Shift go-aligned weights to response-aligned timing per trial
            print("Applying response-aligned weights for resp condition")
            time_axis_go = np.linspace(go_weight_timing[0], go_weight_timing[1], weights_go.shape[-1])
            time_axis_target = np.linspace(timing[0], timing[1], in_data_full.shape[-1])

            weights_go_base = weights_go[..., 0, :] if weights_go.shape[-2] == 1 else weights_go
            rt_array_np = np.array(rt_array)

            # Apply per-channel, per-trial response-aligned weights
            # Data shape: (channels, freq, trials, time)
            for ch_idx in range(in_data_full.shape[0]):
                interp_func = interp1d(time_axis_go, weights_go_base[ch_idx], kind='linear',
                                       axis=-1, bounds_error=False, fill_value=0.0)
                for trial_idx in range(in_data_full.shape[2]):
                    rt_val = rt_array_np[ch_idx, trial_idx]
                    if np.isnan(rt_val) or rt_val <= 0:
                        weights_resp_aligned = interp_func(time_axis_target)
                    else:
                        weights_resp_aligned = interp_func(time_axis_target + rt_val)
                    trial_data = in_data_full[ch_idx, :, trial_idx, :]
                    weighted_preserve_stats(trial_data, weights_resp_aligned)
                    in_data_full[ch_idx, :, trial_idx, :] = trial_data
        else:
            weighted_preserve_stats(in_data_full, weights_go)

        in_data = np.nanmean(in_data_full, axis=1)

        print(f"Constructed RT array with shape={getattr(rt_array, 'shape', None)}")

        # # perform linear regression rt prediction
        y = np.array(rt_array)
        valid_idx = ~np.isnan(in_data[..., 0]) & ~np.isnan(y)
        # get significance with permutation test

        # results = []
        x = np.array(in_data, dtype='f2')[valid_idx]
        y = np.broadcast_to(np.array(rt_array, dtype='f2')[valid_idx, None], x.shape)
        results = time_perm_cluster(x, y, 0.05,
                stat_func=ridge_nd,
                permutation_type='pairings',
                vectorized=True,
                n_perm=n_permutations,
                tails=1,
                axis=0)

        corr, beta, intercept = ridge_nd(x,y, 0, return_params=True)
        corrs.append(corr)
        betas.append(beta)
        intercepts.append(intercept)
        sigs.append(results[0])

    #     # %% univariate analysis of regression with rt (per component)
    #     x = np.array(in_data, dtype='f2')
    #     y = np.broadcast_to(np.array(rt_array, dtype='f2')[..., None], x.shape)
    #
    #     uv_r2, uv_betas, uv_intercepts = ridge_nd(
    #         x, y,
    #         axis=1,
    #         return_params=True,
    #     )
    #
    #     uv_results = time_perm_cluster(
    #         x, y,
    #         0.05,
    #         tails=1,
    #         axis=-2,
    #         stat_func=ridge_nd,
    #         permutation_type='pairings',
    #         vectorized=True,
    #         n_perm=n_permutations,
    #         ignore_adjacency=0
    #     )
    #     uv_sig = uv_results[0]
    #
    #     uv_betas_all.append(uv_betas)
    #     uv_r2_all.append(uv_r2)
    #     uv_sig_all.append(uv_sig)
    #     uv_labels_all.append(rt_array.labels[0])
    # Get time axis for in_data time dimension
    time_axis_in_data = np.broadcast_to(np.linspace(timing[0], timing[1],
                                    in_data.shape[-1])[None], (n_components[0], in_data.shape[-1]))
    corrs = np.array(corrs)
    betas = np.array(betas)
    sigs = np.array(sigs)
    plt.figure(figsize=(8, 4))
    # plt.plot(time_axis_in_data, betas, label='RT Prediction Coefficient')
    ax = plt.gca()
    for i in range(n_components[0]):
        ax.plot(time_axis_in_data[i], corrs[i], label=names[i], color=colors[i])
    # plt.plot(time_axis_in_data.T, corrs.T, colors=colors)
    # plt.plot(time_axis_in_data, intercepts, label='Intercept')
    ax = plt.gca()
    plot_horizontal_bars(ax, sigs)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('R^2')
    plt.title(
        f'Reaction Time Prediction R^2 over Time ({group} channels)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'rt_prediction_{group}.png'))

    # # %% univariate analysis plots (all components on same axes)
    # plt.figure(figsize=(8, 4))
    # ax = plt.gca()
    # for i in range(n_components[0]):
    #     plot_dist(uv_betas_all[i], 0, label=names[i], color=colors[i],
    #               times=timing, ax=ax)
    # plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Regression Coefficient')
    # plt.title(f'Univariate Reaction Time Prediction Coefficients over Time ({group} channels, {cond})')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(log_dir, f'rt_prediction_univariate_{group}_{cond}.png'))
    #
    # plt.figure(figsize=(8, 4))
    # ax = plt.gca()
    # for i in range(n_components[0]):
    #     plot_dist(uv_r2_all[i], 0, label=names[i], color=colors[i],
    #               times=timing, ax=ax)
    # plt.xlabel('Time (s)')
    # plt.ylabel('R^2')
    # plt.title(f'Univariate Reaction Time Prediction R^2 over Time ({group} channels, {cond})')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(log_dir, f'rt_prediction_univariate_r2_{group}_{cond}.png'))
    #
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # axs_flat = axs.flatten()
    # for i in range(n_components[0]):
    #     ax = axs_flat[i]
    #     sig = uv_sig_all[i]
    #     ax.imshow(sig, cmap='viridis', aspect='auto')
    #     ax.set_title(names[i])
    #     ax.set_ylabel('Channel')
    #     ax.set_xlabel('Time (s)')
    #     ax.set_yticks(range(0, sig.shape[0], 20))
    #     ax.set_yticklabels(uv_labels_all[i][0:-1:20], rotation=45)
    #     ax.set_xticks(range(0, sig.shape[-1], 25))
    #     ax.set_xticklabels(np.arange(timing[0], timing[1], 0.25))
    # for j in range(n_components[0], len(axs_flat)):
    #     axs_flat[j].axis('off')
    # fig.suptitle(f'Univariate Reaction Time Prediction Significance over Time ({group} channels, {cond})')
    # fig.tight_layout()
    # plt.savefig(os.path.join(log_dir, f'rt_prediction_univariate_sig_{group}_{cond}.png'))

