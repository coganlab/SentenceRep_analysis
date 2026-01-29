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
        np.full(in_data.shape[:2], np.nan),
        in_data.labels[:2]
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

for cond, timing in zip(('go_ls', 'resp'), ((-0.5, 1.5), (-1, 1))):
    print(f"Processing condition: {cond} with timing {timing}")
    # sub-select data to analyze
    in_data = np.nanmean(zscores[cond][:, labels[0]], 2).combine((0, 2)).dropna()

    # create rt matrix that matches the data
    print("Applying response-aligned weights for resp condition")

    # Map subset channel indices to subjects
    unique_subjects = get_unique_subjects(labels[0], zscores.labels[2].tolist())

    # Get tokens from neural data
    # Data structure: (tokens, channels, freq, trials, time)
    data_tokens = ['heat', 'hoot', 'hot', 'hut']

    print(f"Number of trials from labels[1]: {len(in_data.labels[1])}")

    # Get time axis for in_data time dimension
    time_axis_in_data = np.linspace(timing[0], timing[1], in_data.shape[-1])

    rt_array = make_rt_array(rt_dict, trial_df_dict, labels, unique_subjects, exclude)

    print(f"Constructed RT array with shape={getattr(rt_array, 'shape', None)}")

    # perform linear regression rt prediction
    lin_model = Ridge(alpha=1, positive=False, tol=1e-6)
    betas = []
    corrs = []
    y = np.array(rt_array)
    valid_idx = ~np.isnan(in_data[..., 0]) & ~np.isnan(y)

    for t in range(in_data.shape[-1]):
        X = np.array(in_data[:, :, t])
        lin_model.fit(X[valid_idx][:, None], y[valid_idx])
        y_pred = lin_model.predict(X[valid_idx][:, None])
        corr = np.corrcoef(y[valid_idx], y_pred)[0, 1]
        print(f"Time {time_axis_in_data[t]:.3f}s: RT prediction correlation: {corr:.4f}")
        betas.append(lin_model.coef_[0])
        corrs.append(corr)

    betas = np.array(betas)  # shape: (time, channels)
    corrs = np.array(corrs)

    # get significance with permutation test
    n_permutations = 1000
    threshold = None  # use default threshold

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
    # for t in range(in_data.shape[-1]):
    # #     result = permutation_test(
    # #         (np.array(in_data[..., t])[valid_idx], y[valid_idx]),
    # #         stat_func,
    # #         permutation_type='pairings',
    # #         vectorized=False,
    # #         n_resamples=n_permutations,
    # #         alternative='greater',
    # #         axis=0,
    # #     )
    # #     results.append(result)
    # #     print(f"Time {time_axis_in_data[t]:.3f}s: permutation test p-value: {result.pvalue:.4f}")


    corrs, betas, intercepts = ridge_nd(x,y, 0, return_params=True)
    plt.figure(figsize=(8, 4))
    # plt.plot(time_axis_in_data, betas, label='RT Prediction Coefficient')
    plt.plot(time_axis_in_data, corrs)
    # plt.plot(time_axis_in_data, intercepts, label='Intercept')
    ax = plt.gca()
    plot_horizontal_bars(ax, results[0][None])
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('R^2')
    plt.title(
        f'Reaction Time Prediction R^2 over Time ({group} channels)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'rt_prediction_{group}.png'))


    # univariate analysis of regression with rt (per subject)
    # lin_model = LinearRegression()
    betas = np.zeros(in_data.shape[0::2])
    corrs = np.zeros((len(unique_subjects), in_data.shape[2]))

    for i, sub in enumerate(unique_subjects.keys()):
        chan_idx = [r.startswith(sub) for r in rt_array.labels[0]]
        ys = np.array(rt_array)[chan_idx]
        xs = np.array(in_data)[chan_idx]
        good_trials = ~np.any(np.isnan(ys), axis=0) & ~np.any(
            np.isnan(xs), axis=(0, 2))
        y = np.broadcast_to(ys[0, good_trials, None], xs[0, good_trials].shape)
        print(f"Subject {sub}: Performing univariate RT regression on "
              f"{np.sum(good_trials)} good trials and {np.sum(chan_idx)} channels")
        outs = ridge_nd(
            xs[:, good_trials, :].astype(float), y,
            axis=-2, features_axis=0, return_params=True)
        betas[chan_idx] = outs[1].T
        corrs[i, :] = outs[0]


    # univariate analysis of regression with rt (per subject)
    # lin_model = LinearRegression()
    # betas = np.zeros(in_data.shape[0::2])
    # r2 = np.zeros(in_data.shape[0::2])
    # sig = np.zeros(in_data.shape[0::2])

    # for i, r in enumerate(rt_array.labels[0]):
    #     ys = np.array(rt_array)[i]
    #     xs = np.array(in_data)[i]
    #     good_trials = ~np.isnan(ys) & ~np.any(np.isnan(xs), axis=1)
    #     y = np.broadcast_to(ys[good_trials, None], xs[good_trials].shape)
    #     x = xs[good_trials].astype("f4")
    #     print(f"Channel {r}: Performing univariate RT regression on "
    #           f"{np.sum(good_trials)} good trials")
    #     outs = ridge_nd(x, y, axis=0, return_params=True)
    #     betas[i] = outs[1]
    #     r2[i] = outs[0]
    #     result = time_perm_cluster(x, y, 0.05,
    #         stat_func=ridge_nd,
    #         permutation_type='pairings',
    #         vectorized=True,
    #         n_perm=n_permutations,
    #         tails=1,
    #         axis=0)
    #     sig[i] = result[0]

    x = np.array(in_data, dtype='f2')
    y = np.broadcast_to(np.array(rt_array, dtype='f2')[..., None], x.shape)

    r2, betas, intercepts = ridge_nd(
        x, y,
        axis=1,
        return_params=True,
    )

    results = time_perm_cluster(
        x, y,
        0.05,
        tails=1,
        axis=-2,
        stat_func=ridge_nd,
        permutation_type='pairings',
        vectorized=True,
        n_perm=n_permutations,
        ignore_adjacency=0
    )
    sig = results[0]


    plt.figure(figsize=(8, 4))
    plot_dist(betas, 0, label = 'RT Prediction Coefficient',
              times = timing, ax=plt.gca())
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Regression Coefficient')
    plt.title(f'Univariate Reaction Time Prediction Coefficients over Time ({group} channels, {cond})')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'rt_prediction_univariate_{group}_{cond}.png'))

    plt.figure(figsize=(8, 4))
    plot_dist(r2, 0,
              times = timing, ax=plt.gca())
    plt.xlabel('Time (s)')
    plt.ylabel('R^2')
    plt.title(f'Univariate Reaction Time Prediction R^2 over Time ({group} channels, {cond})')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'rt_prediction_univariate_r2_{group}_{cond}.png'))

    plt.figure(figsize=(8, 4))
    plt.imshow(sig, cmap='viridis', aspect='auto')
    plt.ylabel('Channel')
    plt.yticks(range(0, sig.shape[0], 20), rt_array.labels[0][0:-1:20], rotation=45)
    plt.xlabel('Time (s)')
    plt.xticks(range(0, sig.shape[-1], 25), np.arange(timing[0], timing[1], 0.25))
    plt.title(f'Univariate Reaction Time Prediction Significance over Time ({group} channels, {cond})')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'rt_prediction_univariate_sig_{group}_{cond}.png'))

