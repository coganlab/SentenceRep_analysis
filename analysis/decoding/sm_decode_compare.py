import os
import json
import numpy as np
import torch

from functools import partial
import pandas as pd

from sklearn import set_config
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA, FactorAnalysis as FA
from tqdm import tqdm

from ieeg.io import get_data
from analysis.grouping import group_elecs
from analysis.load import load_data, load_spec
from ieeg.decoding.decode import Decoder
from ieeg.decoding.models import PcaLdaClassification, PcaEstimateDecoder
from ieeg.arrays.reshape import sliding_window_view
from analysis.decoding.utils import get_scores


def weighted_preserve_stats(data, weights, axis=None):
    """
    Multiply data along the specified axis by normalized weights to preserve
    mean scaling. See words_factor.py for original reference.

    Parameters
    ----------
    data : np.ndarray or LabeledArray
        Input data to be weighted in-place.
    weights : np.ndarray
        Weight vector per-channel.
    axis : int | None
        Axis to apply weights along. If None, multiply entire array.
    """
    w = weights / np.mean(weights)
    if axis is None:
        data *= w
    else:
        data *= w.reshape([1 if i != axis else -1 for i in range(data.ndim)])


def compute_accuracy_from_cm(cm: np.ndarray) -> float:
    """Return balanced accuracy from confusion matrix."""
    n_classes = cm.shape[-1]
    return np.mean(cm.T[np.eye(n_classes).astype(bool)].T,
                   axis=-1)


if __name__ == "__main__":
    set_config(enable_metadata_routing=True)

    # ----- Configuration -----
    # exclude = [
    #     "D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
    #     "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
    #     "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
    #     "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
    #     "D0027-LPIF10"
    # ]
    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "workspace", "CoganLab") if 'SLURM_ARRAY_TASK_ID' in os.environ else os.path.join(HOME, "Box", "CoganLab")
    folder = 'stats_freq_hilbert'
    # Trials axis: prior scripts used obs_axs=2 for get_scores; keep consistent
    obs_axs = 2
    # Frequency axis for zscores is axis=3 (based on prior usage)
    freq_ax = 3
    # Channel axis is 2
    ch_ax = 2

    # Data augmentation grid (independent of model selections)
    augmentation_grid = list(ParameterGrid({
        'variant': ['weighted'],  # how to use SliceTCA components
        'frequency_averaged': [#False,
                               # 1/3,
                               # 1/6,
            # 1/10,
            True],#, True],     # average across frequency axis
        'threshold': [0.05],                     # threshold for channel subset (relative to mean W)
        'time_slice': [slice(None, 175),
            #slice(0, None, 2), slice(0,None,4),
            #slice(50, 150),
            #slice(50, 150, 2),
            # slice(0, 50)
            # slice(75, 125),
            # slice(75, 125, 2)
        ],            # time slice to use (if any)
    }))

    # Model grids (each model with its own hyperparameters)
    model_grids = {
        # 'lda': list(ParameterGrid({
        #     'explained_variance': [0.3,0.4,0.5,0.6,0.7, 0.8, 0.85, 0.9, 0.95],
        # })),
        'svm': list(ParameterGrid([
            {'pca': [PCA, FA], 'explained_variance': [100], 'kernel': ['linear'], 'C': [1, 0.1, 0.01, 0.001]},
            # {'explained_variance': [0.6, 0.7, 0.8, 0.9], 'gamma': [0.0001, 'auto', 'scale'],
            #  'kernel': ['sigmoid'], 'C': [0.1, 0.01, 0.001]},
        ])),
        # 'nn': list(ParameterGrid({ ... }))  # reserved for future neural net models
    }

    # Word decoding classes; conditions are evaluated in pairs
    word_categories = {'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3}
    # conds = list(map(list, list(combinations(['aud_ls', 'aud_lm', 'aud_jl'], 2)) +
    #                  list(combinations(['go_ls', 'go_lm', 'go_jl'], 2))))
    conds = [#['aud_ls', 'aud_lm'],
             ['go_ls', 'go_lm']]

    # ----- Load data -----
    layout = get_data('SentenceRep', root=LAB_root)
    sigs = load_data(layout, folder, 'mask')
    AUD, SM, PROD, sig_chans, delay = group_elecs(
        sigs,
        [s for s in sigs.labels[1]],
        sigs.labels[0]
    )
    zscores = load_data(layout, folder, 'zscore')

    # ----- Load SliceTCA model and channel labels (as in words_factor.py) -----
    # Expect these files in the working directory as per existing scripts
    idx_name = 'SM'
    if os.path.exists(f'{idx_name}_chns.pkl'):
        import pickle
        labels = pickle.load(open(f'{idx_name}_chns.pkl', 'rb'))
    else:
        # Fallback: build labels from spec
        labels = load_spec(idx_name, list({c for pair in conds for c in pair}),
                           layout, folder=folder, min_nan=1, n_jobs=-2)[-2]
    idx_num = [zscores.find(l, ch_ax) for l in labels[0]]

    state = torch.load('model_SM3_freq.pt')
    import slicetca
    model_tca = slicetca.core.SliceTCA(
        dimensions=state['vectors.0.0'].shape[1:] + state['vectors.0.1'].shape[1:],
        ranks=(state['vectors.0.0'].shape[0], 0, 0, 0),
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
    model_tca.load_state_dict(state)
    W, H = model_tca.get_components(numpy=True)[0]
    n_components = W.shape[0]
    n_components = 1

    # Results structure
    results = []

    # Decoder evaluation args (non-windowed: simply omit window/step)
    common_kwargs = {
        'obs_axs': obs_axs,
        'normalize': 'true',
        'n_jobs': 1,
        'average_repetitions': True,
        'on_gpu': True,
        'shuffle': False,
        'which': -2,  # keep consistent with prior scripts
        # 'window': 20,
        # 'step': 5
    }
    n_splits = 10
    n_repeats = 5

    t = tqdm(desc='overall',
             total=n_components * len(augmentation_grid) *
                   sum(len(grid) for grid in model_grids.values()) * n_splits
                   * n_repeats * len(conds))
    # Iterate augmentations (data) first, then models (algorithms)
    for aug in augmentation_grid:
        variant = aug['variant']
        freq_avg = aug['frequency_averaged']
        component_threshold = float(aug['threshold'])
        time_slice = aug['time_slice']

        for model_name, grid in model_grids.items():
            for params in grid:
                # Build model per-params
                if model_name == 'lda':
                    decode_model = PcaLdaClassification(
                        explained_variance=params['explained_variance'],
                        da_type='lda')
                    common_kwargs['on_gpu'] = True
                    common_kwargs['n_jobs'] = 1
                elif model_name == 'svm':
                    decode_model = PcaEstimateDecoder(
                        explained_variance=params['explained_variance'],
                        clf_params={'kernel': params['kernel']}
                    )
                    common_kwargs['on_gpu'] = False
                    common_kwargs['n_jobs'] = 10
                else:
                    continue

                decoder = Decoder(word_categories, n_splits, n_repeats, 2,
                                  'train', model=decode_model)
                decoder.t = t

                # Loop components and aggregate accuracy
                comp_scores = {}
                comp_scores_ave = {}
                for i in range(n_components):
                    i = 1

                    # Select channels by threshold relative to mean weight across channels
                    W_subset = np.nonzero(W[i] / W.mean() > component_threshold)[0]
                    if W_subset.size == 0:
                        continue
                    subset = [idx_num[s] for s in W_subset]

                    # Restrict to selected channels (and SM mapping)
                    # labels[0] maps component channel indices to zscores channel label indices
                    in_data = zscores[:, :, subset, ..., time_slice]

                    # Apply weights per condition for the weighted variant
                    if variant == 'weighted':
                        comp_weights = model_tca.construct_single_component(
                            0, i).detach().numpy()[
                            None, W_subset, :, :, None]
                        weights_aud, weights_go = np.split(
                            np.nanmean(comp_weights, axis=2),
                            [175], axis=-1)

                        # Apply to each condition set
                        for c in ['ls', 'lm', 'jl']:
                            weighted_preserve_stats(in_data['aud_' + c],
                                                    weights_aud[..., time_slice])
                            weighted_preserve_stats(in_data['go_' + c],
                                                    weights_go[..., time_slice])

                    # Frequency averaging if requested (axis may be absent)
                    if freq_avg is True or freq_avg == 1:
                        in_data = np.nanmean(in_data, axis=freq_ax,
                                             keepdims=True)
                    elif isinstance(freq_avg, float) and 1 > freq_avg > 1/in_data.shape[freq_ax]:
                        #moving average to partial reduce frequency dimension
                        window = int(round(1 / freq_avg))
                        idx = [slice(None)] * in_data.ndim
                        idx[freq_ax] = slice(0, None, window)
                        in_data = np.nanmean(sliding_window_view(
                            in_data, window_shape=window, axis=freq_ax,
                            subok=True)[tuple(idx)], axis=-1)

                    # Run non-windowed decoding for this component
                    # Using a single group with all selected channels
                    idxs = [list(range(len(subset)))]
                    names = [f"{idx_name}-{aug}-{params}-{i}"]
                    for values in get_scores(in_data, decoder, idxs, conds, names, **common_kwargs):
                        key = '-'.join(decoder.current_job.split('-')[-2:] + [str(i)])
                        comp_scores[key] = np.asarray(values)
                        comp_scores_ave[key] = compute_accuracy_from_cm(
                            np.asarray(values))

                results.append({
                    'model': model_name,
                    **params,
                    **aug,
                    'accuracy_cm': comp_scores,
                    'accuracy_mean': comp_scores_ave
                })

    t.close()

    # create a dataframe from results
    df = pd.json_normalize(results)
    df.sort_values('accuracy_mean.go_ls-go_lm-1', ascending=False, inplace=True)
    print(df.head())
    df_top20 = df.head(20)

    # # Save results alongside script
    name = 'sm_decode_compare_go_svm'
    df.to_csv(f'{name}_results.csv')
    with open(f'{name}_config.json', 'w') as f:
        json.dump({
            'obs_axs': obs_axs,
            'freq_ax': freq_ax,
            'ch_ax': ch_ax,
            'augmentation_grid': augmentation_grid,
            'model_grids': model_grids,
            'word_categories': word_categories,
            'conds': conds,
            'n_splits': n_splits,
            'n_repeats': n_repeats,
        }, f, indent=4)
    df.to_json(f'{name}_results.json', orient='records', indent=4)
    print('Done')


