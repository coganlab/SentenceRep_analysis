# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os
from numpy.lib.stride_tricks import sliding_window_view

from analysis.grouping import group_elecs
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.decoding.decode import Decoder, plot_all_scores
from ieeg.decoding.models import SEEGConformerClassifier, ResNetTokenClassifier, PcaLdaClassification
from ieeg.calc.stats import time_perm_cluster
from analysis.decoding.utils import get_scores
import torch
import slicetca
from functools import partial
from ieeg.io import get_data
from analysis.load import load_data, load_spec
from sklearn import set_config
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

from ieeg.viz.ensemble import plot_dist_bound


set_config(enable_metadata_routing=True)


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
            raise ValueError(f"Weights size ({weights.size}) doesn't match data dimension "
                           f"along axis {axis} ({data.shape[axis]})")

    # Multiply along the specified axis
    if axis is None:
        data *= w
    else:
        w_reshaped = w.reshape([1 if i != axis else -1 for i in range(data.ndim)])
        data *= w_reshaped


def _p(msg, *args):
    """Simple print wrapper for consistent debug output."""
    try:
        print("[words_factor DEBUG] " + msg.format(*args))
    except Exception:
        print("[words_factor DEBUG] " + str(msg))


def get_channel_coordinates(layout, channel_names, folder='stats_freq_hilbert', 
                            coord_frame='mni_tal'):
    """Extract 3D coordinates for channels from BIDS data.
    
    Tries multiple methods:
    1. From raw iEEG files via mne_bids
    2. From epoch files in derivatives folder
    3. Falls back to NaN if none available
    
    Parameters
    ----------
    layout : BIDSLayout
        PyBIDS layout object
    channel_names : list[str]
        List of channel names to get coordinates for
    folder : str
        Derivatives folder name (default: 'stats_freq_hilbert')
    coord_frame : str
        Coordinate frame to use (default: 'mni_tal')
        
    Returns
    -------
    np.ndarray
        Array of shape (n_channels, 3) with x, y, z coordinates in meters.
        Channels not found will have NaN coordinates.
    """
    import mne
    from mne_bids import read_raw_bids, BIDSPath
    from ieeg.io import bidspath_from_layout, DataLoader
    
    # Get coordinates from first available subject/session
    subjects = sorted(layout.get_subjects())
    if not subjects:
        _p("Warning: No subjects found in layout, returning NaN coordinates")
        return np.full((len(channel_names), 3), np.nan)
    
    # Method 1: Try to get from raw iEEG file
    try:
        # Get first subject and session
        subject = subjects[0]
        sessions = layout.get_sessions(subject=subject)
        session = sessions[0] if sessions else None
        
        # Search for raw iEEG file
        kwargs = {'subject': subject, 'datatype': 'ieeg', 'suffix': 'ieeg'}
        if session:
            kwargs['session'] = session
        
        # Try different file extensions
        raw = None
        for ext in ['.vhdr', '.edf', '.bdf', '.set', '.fif']:
            try:
                kwargs['extension'] = ext
                bids_path = bidspath_from_layout(layout, **kwargs)
                raw = read_raw_bids(bids_path, verbose=False)
                break
            except (FileNotFoundError, ValueError, KeyError):
                continue
        
        if raw is None:
            raise FileNotFoundError("Could not find raw iEEG file")
        
        # Get montage and positions
        info = raw.info
        montage = info.get_montage()
        if montage is None:
            raise ValueError("No montage in raw file")
        
        pos_dict = montage.get_positions()['ch_pos']
        _p("Found {} channels in montage", len(pos_dict))
        
    except Exception as e1:
        _p("Method 1 (raw file) failed: {}, trying epoch files", e1)
        # Method 2: Try to get from epoch file in derivatives
        try:
            # Try to load an epoch file to get channel info
            conds_test = list(layout.get_entities()['condition'] if 'condition' in layout.get_entities() else ['aud_ls'])[:1]
            if not conds_test:
                conds_test = ['aud_ls']  # Default condition
            
            loader = DataLoader(layout, {conds_test[0]: (-0.5, 1.5)}, 
                              datatype="zscore", average=False, 
                              derivatives_folder=folder, ext='.h5')
            # Try to load one subject/condition to get channel info
            subject = subjects[0]
            _, _, epoch_dict = loader.load_subject_condition(subject, conds_test[0])
            
            if epoch_dict is None:
                raise ValueError("Could not load epoch data")
            
            # Get first channel's data structure to access MNE object
            # This is tricky - we need to access the underlying MNE object
            # For now, fall back to trying raw files from all subjects
            raise ValueError("Epoch method not fully implemented, trying all subjects")
            
        except Exception as e2:
            _p("Method 2 (epoch file) failed: {}, trying all subjects", e2)
            # Method 3: Try all subjects until we find one with coordinates
            pos_dict = {}
            for subject in subjects:
                try:
                    sessions = layout.get_sessions(subject=subject)
                    session = sessions[0] if sessions else None
                    kwargs = {'subject': subject, 'datatype': 'ieeg', 'suffix': 'ieeg'}
                    if session:
                        kwargs['session'] = session
                    
                    for ext in ['.vhdr', '.edf', '.bdf', '.set', '.fif']:
                        try:
                            kwargs['extension'] = ext
                            bids_path = bidspath_from_layout(layout, **kwargs)
                            raw = read_raw_bids(bids_path, verbose=False)
                            montage = raw.info.get_montage()
                            if montage is not None:
                                pos_dict = montage.get_positions()['ch_pos']
                                _p("Found coordinates from subject {}: {} channels", subject, len(pos_dict))
                                break
                        except (FileNotFoundError, ValueError, KeyError, AttributeError):
                            continue
                    
                    if pos_dict:
                        break
                except Exception:
                    continue
            
            if not pos_dict:
                _p("Warning: Could not extract coordinates from any source, returning NaN")
                return np.full((len(channel_names), 3), np.nan)
    
    # Convert to numpy array matching channel_names order
    coords = np.full((len(channel_names), 3), np.nan)
    for idx, ch_name in enumerate(channel_names):
        # Try exact match first
        if ch_name in pos_dict:
            coords[idx] = pos_dict[ch_name]
        else:
            # Try matching without subject prefix (e.g., 'D0001-LPIF1' -> 'LPIF1')
            ch_short = ch_name.split('-')[-1] if '-' in ch_name else ch_name
            for montage_ch, pos in pos_dict.items():
                montage_short = montage_ch.split('-')[-1] if '-' in montage_ch else montage_ch
                if ch_short == montage_short:
                    coords[idx] = pos
                    break
    
    # Positions are in meters in MNE (keep as meters for model)
    _p("Extracted coordinates for {}/{} channels", np.sum(~np.isnan(coords[:, 0])), len(channel_names))
    return coords


def load_response_times(layout, subjects_list, rt_deriv_folder='rt_hg'):
    """Load response times from rt_hg_map output files.
    
    Parameters
    ----------
    layout : BIDSLayout
        BIDS layout object
    subjects_list : list[str]
        List of subject IDs
    rt_deriv_folder : str
        Name of derivatives folder containing RT data (default: 'rt_hg')
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping subject ID to array of response times
    dict[str, pd.DataFrame]
        Dictionary mapping subject ID to trial DataFrame with tokens
    """
    import pandas as pd
    import glob
    
    rt_dict = {}
    trial_df_dict = {}
    
    rt_dir = os.path.join(layout.root, 'derivatives', rt_deriv_folder)
    
    for subj in subjects_list:
        tsv_path = os.path.join(rt_dir, f"{subj}_rt.tsv")
        if os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep='\t')
            # Filter to Word LS trials with valid response times
            df_filtered = df[(df['trial_type'] == 'Word') & 
                            (df['condition'] == 'LS') &
                            (df['response_time'].notna())].copy()
            # Sort by trial_index to ensure consistent ordering
            df_filtered = df_filtered.sort_values('trial_index').reset_index(drop=True)
            rt_dict[subj] = df_filtered['response_time'].to_numpy()
            trial_df_dict[subj] = df_filtered
        else:
            rt_dict[subj] = None
            trial_df_dict[subj] = None
    
    return rt_dict, trial_df_dict


def create_response_aligned_weights(weights_go, rt_array, time_axis_go, time_axis_resp, sfreq=100.0):
    """Create response-aligned weights by shifting go-aligned weights based on RT.
    
    Parameters
    ----------
    weights_go : np.ndarray
        Go-aligned weights, shape (..., n_timepoints_go) where last axis is time
    rt_array : np.ndarray
        Response times for each trial, shape (n_trials,)
    time_axis_go : np.ndarray
        Time axis for go-aligned data (in seconds, relative to go cue), typically -0.5 to 1.5
    time_axis_resp : np.ndarray
        Time axis for response-aligned data (in seconds, relative to response), typically -1 to 1
    sfreq : float
        Sampling frequency (default: 100.0 Hz)
        
    Returns
    -------
    np.ndarray
        Response-aligned weights, shape (n_trials, ..., n_timepoints_resp)
    """
    
    n_trials = len(rt_array)
    n_timepoints_go = len(time_axis_go)
    n_timepoints_resp = len(time_axis_resp)
    
    # Get original shape without time axis
    weights_shape = weights_go.shape[:-1]
    
    # Create output array: (n_trials, ..., n_timepoints_resp)
    output_shape = (n_trials,) + weights_shape + (n_timepoints_resp,)
    weights_resp_aligned = np.zeros(output_shape)
    
    # For each trial, shift weights backward by RT
    for trial_idx in range(n_trials):
        rt = rt_array[trial_idx]
        if np.isnan(rt) or rt <= 0:
            # If RT is invalid, interpolate go-aligned weights to response-aligned grid without shift
            weights_flat = weights_go.reshape(-1, n_timepoints_go)
            weights_resp_flat = np.zeros((weights_flat.shape[0], n_timepoints_resp))
            for w_idx in range(weights_flat.shape[0]):
                interp_func = interp1d(time_axis_go, weights_flat[w_idx], 
                                     kind='linear', 
                                     bounds_error=False, 
                                     fill_value=0.0)
                weights_resp_flat[w_idx] = interp_func(time_axis_resp)
            weights_resp_aligned[trial_idx] = weights_resp_flat.reshape(weights_shape + (n_timepoints_resp,))
            continue
        
        # Convert response-aligned time to go-aligned time
        # Response time t_resp corresponds to go time t_go = t_resp + rt
        time_go_from_resp = time_axis_resp + rt
        
        # Interpolate weights to response-aligned time grid
        # Flatten non-time dimensions for interpolation
        weights_flat = weights_go.reshape(-1, n_timepoints_go)
        weights_resp_flat = np.zeros((weights_flat.shape[0], n_timepoints_resp))
        
        for w_idx in range(weights_flat.shape[0]):
            # Interpolate each weight vector from go-aligned to response-aligned
            interp_func = interp1d(time_axis_go, weights_flat[w_idx], 
                                 kind='linear', 
                                 bounds_error=False, 
                                 fill_value=0.0)
            weights_resp_flat[w_idx] = interp_func(time_go_from_resp)
        
        # Reshape back to original shape
        weights_resp_aligned[trial_idx] = weights_resp_flat.reshape(weights_shape + (n_timepoints_resp,))
    
    return weights_resp_aligned



if __name__ == '__main__':


    exclude = ["D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
               "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
               "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
               "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
               "D0027-LPIF10"]

    HOME = os.path.expanduser("~")
    _p("HOME directory: {}", HOME)
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
    _p("Log directory: {}", log_dir)
    layout = get_data('SentenceRep', root=LAB_root)
    _p("Loaded layout: {}", getattr(layout, 'root', str(layout)))

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
    _p("Loaded sigs: type={} labels={}", type(sigs), getattr(sigs, 'labels', None))
    AUD, SM, PROD, sig_chans, delay = group_elecs(sigs,
                                                  [s for s in
                                                   sigs.labels[1]
                                                   if s not in exclude],
                                                  sigs.labels[0])
    _p("Grouped electrodes: AUD={} SM={} PROD={} sig_chans={} delay={}", len(AUD), len(SM), len(PROD), len(sig_chans), len(delay))
    idxs = {'SM': sorted(SM), 'AUD': sorted(AUD), 'PROD': sorted(PROD),
            'sig_chans': sorted(sig_chans), 'delay': sorted(delay)}
    # idx = idxs[group]
    zscores = load_data(layout, folder, "zscore")
    _p("Loaded zscores: {}", type(zscores))

    n_components = (4,)
    best_seed = 123457
    window_kwargs = {'window':20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': 1,
                    'average_repetitions': False, 'step': 8}

    # #
    # %% decompose the optimal model

    import pickle
    filename = f"{group}_chns.pkl"
    if os.path.exists(filename):
        labels = pickle.load(open(filename, 'rb'))
        _p("Loaded labels from pickle: {} entries", len(labels[0]) if hasattr(labels, '__len__') else 'unknown')
    else:
        labels = load_spec(group, list(conds_all.keys()), layout, folder=folder,
                           min_nan=1, n_jobs=-2)[-2]
        pickle.dump(labels, open(filename, 'wb'))
        _p("Computed and saved labels to {}", filename)
    state = torch.load('model_SM3_freq.pt')
    _p("Loaded torch state keys: {}", list(state.keys())[:10])
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
    _p("Model loaded into slicetca.SliceTCA; n_components={}", n_components)
    # model[0]
    # model = torch.load('model_sig_chans.pt')
    colors = ['orange', 'k', 'c', 'y', 'm', 'deeppink', 'blue']
    names = ['Auditory', 'WM', 'Motor', 'Visual', 'All', 'Other', 'SM']
    W, H = model.get_components(numpy=True)[0]
    W = np.concatenate([W, W.sum(0, keepdims=True)])
    W = np.concatenate([W, W.max() - W[-1:, :]])
    W = np.concatenate([W, np.ones_like(W[:1, :])])
    H = np.concatenate([H, H.sum(0, keepdims=True)])
    H = np.concatenate([H, H.max() - H[-1:, :]])
    H = np.concatenate([H, np.ones_like(H[:1, :])])
    _p("Extracted components W shape={} H shape={}", getattr(W, 'shape', None), getattr(H, 'shape', None))
    # names = ['Auditory', 'WM', 'Motor', 'Visual']
    # conds = ['ls', 'lm', 'jl']
    # names = ['Auditory', 'Visual', 'WM', 'Motor']
    # fig, ax = plt.subplots(1,3)
    # for j, cond in enumerate(conds):
    #     for i in range(n_components[0]):
    #         subset = np.nonzero(W[i] / W.mean() > 0.05)[0]
    #         # subset = np.nonzero(W[i]/W.sum(0) > 0.5)[0]
    #         # subset = np.nonzero(W[i] == np.max(W, 0))[0]
    #         in_data = zscores[:, :, [labels[0][s] for s in subset]]
    #         print(f"{names[i]} component, {len(subset)} channels")
    #         weights = model.construct_single_component(0, i).detach().numpy()[
    #             subset]
    #         weights_aud = np.nanmean(weights[None, ..., None, :200], axis=2)
    #         weights_go = np.nanmean(weights[None, ..., None, 200:], axis=2)
    #         weighted_preserve_stats(in_data['go_' + cond], weights_go)
    #         plot_dist(np.nanmean(in_data['go_' +cond].__array__(), axis=(0, 2, 3)), ax=ax[j], color=colors[i], label=names[i], times=(-0.5, 1.5))
    #     ax[j].set_title(f'go_{cond}')
    #     ax[j].set_xlabel('Time (s)')
    # ax[0].ylabel('Z-Score')
    # plt.legend()
    # raise RuntimeError("stop")

    # %% Time Sliding decoding for word tokens
    # decode_model = PcaLdaClassification(explained_variance=0.8, da_type='lda')
                                        # PCA_kwargs={
    # #     # 'svd_solver': 'full',
    # #     # 'whiten': True
    # })
    # decode_model = PcaEstimateDecoder(96,
    #                                   # pca=FA,
    #                                   clf_params={'max_iter': -1,
    #                                               'kernel': 'sigmoid',
    #                                               'C': 1e-3,
    #                                               'gamma': 1e-5,
    #                                               'tol': 1e-5},)
    
    # Extract channel coordinates once for all channels
    # Get all channel names from zscores
    all_channel_names = list(zscores.labels[2]) if hasattr(zscores, 'labels') and len(zscores.labels) > 2 else []
    _p("Extracting coordinates for {} channels", len(all_channel_names))
    all_channel_coords = get_channel_coordinates(layout, all_channel_names, folder=folder)
    _p("Extracted coordinates: shape={}, valid={}", all_channel_coords.shape, np.sum(~np.isnan(all_channel_coords[:, 0])))
    
    # Create decoder model factory function that accepts channel coordinates
    def create_decode_model(channel_coords=None):
        """Create SEEGConformerClassifier with optional channel coordinates."""
        # return SEEGConformerClassifier(
        #     d_model=128,
        #     nhead_time=8,
        #     depth_time=2,
        #     nhead_space=8,
        #     depth_space=2,
        #     kernel_size=9,
        #     dropout=0.2,
        #     max_epochs=200,
        #     lr=1e-3,
        #     batch_size=128,
        #     device='auto',
        #     oversample=True,
        #     alpha=1.0,
        #     channel_positions=channel_coords
        # )
        return

    decode_model = ResNetTokenClassifier(
            base='resnet18',
            pretrained=False,
            dropout=0.2,
            max_epochs=200,
            lr=1e-3,
            batch_size=128,
            device='auto',
            oversample=True,
            alpha=1.0,
            use_amp=True,
            early_stopping=True,
            es_patience=25,
            es_threshold=1e-3,
            es_load_best=True,
            lr_schedule='plateau',
            lr_factor=0.5,
            lr_patience=10,
            lr_min_lr=1e-6,
            lr_threshold=1e-3,
            lr_cooldown=2,
            optimizer_weight_decay=3e-4,
            label_smoothing=0.05,
            class_weight=None  # or 'balanced' if labels are imbalanced
        )
    
    # Decoder will be created per component with appropriate channel coordinates
    true_scores = {}
    shuffle_scores = {}
    # colors = ['orange', 'k', 'c', 'y', 'm', 'deeppink',
    #           'darkorange', 'lime', 'blue', 'red', 'purple'][:n_components[0]]
    freq_avg = True

    # real_names = ['Auditory', 'Visual', 'WM', 'Motor']
    assert W.shape[1] == len(labels[0])
    _p("Assertion passed: W.shape[1] == len(labels[0]) -> {} == {}", W.shape[1], len(labels[0]))
    idx = [zscores.find(c, 2) for c in
           labels[0]]
    _p("Index mapping created, sample idxs: {}", idx[:5])
    idxs = {c: idx for c in colors}
    window_kwargs = {'window': 20, 'obs_axs': 2, 'normalize': 'true',
                     'n_jobs': 1, 'oversample': True,
                     'average_repetitions': False, 'step': 5,
                     # 'parameter_grid': {'pca__n_components': (1, 96),
                     #                    'classifier__C': (1e-4, 1e-1),
                     #                    'classifier__gamma': (1e-6, 1e-1),
                     #                    'classifier__tol': (1e-5, 1e-1)
                     #                    }
                     }
    # conds = list(map(list, list(combinations(['aud_ls', 'aud_lm', 'aud_jl'], 2))
    #                   + list(
    #              combinations(['go_ls', 'go_lm', 'go_jl'], 2))))
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'],
             'resp']
    # colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]]
    # raise RuntimeError('stop')
    suffix ='_zscore_weighted_words29'
    n_classes = 4
    true_name = 'true_scores' + suffix
    aud_len = 175
    wh = -2
    on_cupy = False

    baseline = 1 / n_classes
    backend = "threading"

    # Load response times for response-aligned weighting
    subjects_list_rt = list(set(s.split('-')[0] for s in sigs.labels[1] if s not in exclude))
    rt_dict, trial_df_dict = load_response_times(layout, subjects_list_rt)
    _p("Loaded RT data for {} subjects", sum(1 for v in rt_dict.values() if v is not None))
    
    # Time axes for weight alignment
    # Sampling rate: 100 Hz
    # Go-aligned data: -0.5 to 1.5 seconds (200 samples at 100 Hz)
    # Response-aligned data: -1 to 1 seconds (200 samples at 100 Hz)
    # aud_len = 175 samples is the auditory period
    # Go period weights start after aud_len: from index 175 to 199 (25 samples)
    # But weights_go_base covers the full go period, so we need the full go time axis
    sfreq = 100.0  # Sampling frequency in Hz
    time_start_go = -0.5
    time_end_go = 1.25
    n_timepoints_total = 175  # Total samples before cropping
    n_timepoints_go = n_timepoints_total - aud_len  # Go period samples
    # Time axis for go period only (after aud_len)
    time_axis_go = np.linspace(time_start_go + (aud_len / sfreq), time_end_go, n_timepoints_go)
    # Time axis for response-aligned data
    time_start_resp = -1.0
    time_end_resp = 1.0
    n_timepoints_resp = 175  # Response-aligned data has 200 samples
    time_axis_resp = np.linspace(time_start_resp, time_end_resp, n_timepoints_resp)

    if not os.path.exists(true_name + '.npz'):
        with joblib.parallel_backend(backend):
            for i in range(n_components[0] + 2):
                _p("Starting loop for component {} of {}", i, n_components[0])
                subset = np.nonzero(W[i] / W.mean() > 0.05)[0]
                # subset = np.nonzero(W[i]/W.sum(0) > 0.5)[0]
                # subset = np.nonzero(W[i] == np.max(W, 0))[0]
                in_data = zscores[:,:,[labels[0][s] for s in subset], ..., :aud_len]

                _p("{ } component, {} channels", names[i], len(subset))
                
                # Extract coordinates for this component's channel subset
                subset_channel_names = [labels[0][s] for s in subset]
                subset_coords = []
                for ch_name in subset_channel_names:
                    # Find this channel in all_channel_names and get its coordinates
                    try:
                        ch_idx = all_channel_names.index(ch_name)
                        subset_coords.append(all_channel_coords[ch_idx])
                    except (ValueError, IndexError):
                        # Channel not found, use NaN coordinates
                        subset_coords.append([np.nan, np.nan, np.nan])
                subset_coords = np.array(subset_coords)  # (n_channels, 3)
                
                # # Only use coordinates if we have valid ones (not all NaN)
                # if np.any(~np.isnan(subset_coords)):
                #     _p("Using coordinates for {}/{} channels in component {}",
                #        np.sum(~np.isnan(subset_coords[:, 0])), len(subset_coords), i)
                #     # Create decoder with coordinates for this component
                #     decode_model = create_decode_model(channel_coords=subset_coords)
                # else:
                #     _p("No valid coordinates found for component {}, using model without coordinates", i)
                #     decode_model = create_decode_model(channel_coords=None)
                
                # Create decoder with this component's model
                decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                                  10, 5, 2, 'train', model=decode_model)
                # weights = W[None, i, subset, None, None, None]
                if i < n_components[0]:
                    weights = model.construct_single_component(0, i).detach().numpy()[subset]
                elif i == n_components[0]:
                    weights = model.construct().detach().numpy()[subset]
                else:
                    ls = np.nanmean(in_data[('aud_ls', 'go_ls'),].combine((0, 5)), axis=(0, 3))
                    lm = np.nanmean(in_data[('aud_lm', 'go_lm'),].combine((0, 5)), axis=(0, 3))
                    jl = np.nanmean(in_data[('aud_jl', 'go_jl'),].combine((0, 5)), axis=(0, 3))
                    weights = np.stack([l.__array__() for l in (ls, lm, jl)], axis=1) - model.construct().detach().numpy()[subset]

                _p("Weights shape: {}", getattr(weights, 'shape', None))
                weights_aud = np.nanmean(weights[None, ..., None, :aud_len], axis=2)
                weights_go = np.nanmean(weights[None, ..., None, aud_len:], axis=2)
                
                # Apply auditory weights (unchanged)
                for c in ['ls', 'lm', 'jl']:
                    _p("Applying weighted_preserve_stats for aud cond {}", c)
                    weighted_preserve_stats(in_data['aud_' + c], weights_aud)
                    _p("Applying weighted_preserve_stats for go cond {}", c)
                    weighted_preserve_stats(in_data['go_' + c], weights_go)
                
                # Apply response-aligned weights to response-aligned data
                if 'resp' in in_data.labels[0]:
                    _p("Applying response-aligned weights for resp condition")
                    resp_data = in_data['resp']
                    
                    # Extract unique subjects from subset channels
                    subset_channel_identifiers = [labels[0][s] for s in subset]
                    all_channel_labels = zscores.labels[2] if hasattr(zscores, 'labels') and len(zscores.labels) > 2 else []
                    
                    # Create mapping from channel identifier to index in all_channel_labels
                    ch_id_to_idx = {ch_id: idx for idx, ch_id in enumerate(all_channel_labels)}
                    
                    # Map subset channel indices to subjects
                    unique_subjects = {}
                    for subset_ch_idx, ch_identifier in enumerate(subset_channel_identifiers):
                        global_ch_idx = ch_id_to_idx.get(ch_identifier)
                        if global_ch_idx is None:
                            _p("Warning: Channel identifier '{}' not found in all_channel_labels, skipping", ch_identifier)
                            continue
                        
                        ch_name = all_channel_labels[global_ch_idx]
                        subj = ch_name.split('-')[0] if '-' in str(ch_name) else str(ch_name)
                        if subj not in unique_subjects:
                            unique_subjects[subj] = []
                        unique_subjects[subj].append(subset_ch_idx)
                    
                    # Get tokens from neural data
                    # Data structure: (tokens, channels, freq, trials, time)
                    data_tokens = None
                    n_trials_data = None
                    if hasattr(resp_data, 'labels') and len(resp_data.labels) > 0:
                        labels_0 = resp_data.labels[0]
                        if hasattr(labels_0, '__len__') and len(labels_0) <= 10:
                            sample_labels = [str(l).lower() for l in list(labels_0)[:4]]
                            if any(tok in ['heat', 'hoot', 'hot', 'hut'] for tok in sample_labels):
                                data_tokens = labels_0
                                _p("Resp data tokens from labels[0]: {}", list(data_tokens))
                        
                        if len(resp_data.labels) > 3:
                            n_trials_data = len(resp_data.labels[3])
                            _p("Number of trials from labels[3]: {}", n_trials_data)
                    
                    # Get time axis for resp_data time dimension
                    n_timepoints_resp_data = resp_data.shape[4]  # Last dimension is time
                    time_axis_resp_data = np.linspace(time_start_resp, time_end_resp, n_timepoints_resp_data)
                    
                    # Iterate over unique subjects
                    for subj, subset_ch_indices in unique_subjects.items():
                        if subj in exclude:
                            continue
                        
                        # Check if any channel for this subject is excluded
                        subj_excluded = False
                        for subset_ch_idx in subset_ch_indices:
                            ch_identifier = subset_channel_identifiers[subset_ch_idx]
                            if ch_identifier in exclude:
                                subj_excluded = True
                                break
                        if subj_excluded:
                            continue
                        
                        trial_df = trial_df_dict.get(subj)
                        rt_array = rt_dict.get(subj)
                        
                        if rt_array is None or len(rt_array) == 0 or trial_df is None:
                            # Raise error if whole subject is missing RT data
                            raise ValueError(f"Subject {subj}: Missing RT data for response-aligned weighting. "
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
                                rt_map[(token, trial_idx)] = rt_val
                        
                        _p("Subject {}: Created RT map with {} entries from {} tokens for resp data", 
                           subj, len(rt_map), len(rt_by_token))
                        
                        # Verify we have token information
                        if data_tokens is None:
                            raise ValueError(f"Subject {subj}: Cannot match trials - token information not available in resp data")
                        
                        if n_trials_data is None:
                            n_trials_data = resp_data.shape[3]  # Dimension 3 is trials
                        
                        # Apply response-aligned weights per (token, trial) combination
                        for token_idx, token in enumerate(data_tokens):
                            token_str = str(token).strip().lower()
                            
                            for trial_idx in range(n_trials_data):
                                rt_val = rt_map.get((token_str, trial_idx))
                                
                                # Set trial to NaN if RT is missing or NaN
                                if rt_val is None or np.isnan(rt_val) or rt_val <= 0:
                                    # Set this trial's data to NaN
                                    for subset_ch_idx in subset_ch_indices:
                                        token_trial_ch_data = resp_data[token_idx:token_idx+1, subset_ch_idx:subset_ch_idx+1, :, trial_idx:trial_idx+1, :]
                                        token_trial_ch_data[...] = np.nan
                                    continue
                                
                                # This trial has valid RT, apply response-aligned weights
                                # Response-aligned time t_resp corresponds to go-aligned time t_go = t_resp + RT
                                # So we shift the go-aligned weights by RT to align with response
                                time_go = np.linspace(time_start_go, time_end_go, n_timepoints_total)
                                
                                for subset_ch_idx in subset_ch_indices:
                                    # Get base go-aligned weights for this channel
                                    ch_weights_base = weights_go[:, subset_ch_idx]
                                    
                                    # Interpolate go-aligned weights to response-aligned time grid using RT shift
                                    interp_func = interp1d(time_go, ch_weights_base,
                                                         kind='linear', bounds_error=False, fill_value=0.0)
                                    weights_resp_aligned = interp_func(time_go + rt_val)
                                    
                                    # Apply to this (token, trial)'s data for this channel
                                    token_trial_ch_data = resp_data[token_idx, subset_ch_idx, :, trial_idx:trial_idx+1, :]
                                    
                                    if token_trial_ch_data.shape[-1] != weights_resp_aligned.shape[-1]:
                                        _p("Error: Weight size ({}) doesn't match data time dimension ({}) for token {}, trial {}, channel {}", 
                                           len(weights_resp_aligned), token_trial_ch_data.shape[4], token_str, trial_idx, subset_ch_idx)
                                        continue
                                    
                                    weighted_preserve_stats(token_trial_ch_data[None], weights_resp_aligned, axis=None)

                # Validate data structure after weight applications
                # Check for any suspiciously large dimensions that might indicate corruption
                max_reasonable_dim = 1e6
                for cond in conds:
                    try:
                        cond_data = in_data[cond]
                        if hasattr(cond_data, 'shape'):
                            for dim_size in cond_data.shape:
                                if dim_size > max_reasonable_dim:
                                    _p("Error: Suspiciously large dimension detected in condition {}: {}", 
                                       cond, cond_data.shape)
                                    raise ValueError(f"Data dimension too large in condition {cond}: {cond_data.shape}. "
                                                   f"This may indicate data corruption from weight application.")
                    except (KeyError, IndexError):
                        # Condition doesn't exist in data, skip validation
                        continue

                # Frequency averaging if requested (axis may be absent)
                if freq_avg is True or freq_avg == 1:
                    in_data = np.nanmean(in_data, axis=-3,
                                         keepdims=True)
                elif isinstance(freq_avg, float) and 1 > freq_avg > 1 / \
                        in_data.shape[-3]:
                    # moving average to partial reduce frequency dimension
                    window = int(round(1 / freq_avg))
                    idx = [slice(None)] * in_data.ndim
                    idx[-3] = slice(0, None, window)
                    in_data = np.nanmean(sliding_window_view(
                        in_data, window_shape=window, axis=-3,
                        subok=True)[tuple(idx)], axis=-1)

                # weighted_preserve_stats(in_data['resp'], W[i, subset], 1)
                # weighted_preserve_stats(in_data, weights, 2)
                
                # Validate data dimensions before passing to decoder
                # Check if any dimension is suspiciously large (likely indicates corruption)
                max_reasonable_dim = 1e6  # Reasonable upper limit for any dimension
                for cond in conds:
                    try:
                        cond_data = in_data[cond]
                        if hasattr(cond_data, 'shape'):
                            if any(d > max_reasonable_dim for d in cond_data.shape):
                                _p("Warning: Suspiciously large dimension detected in condition {}: {}", 
                                   cond, cond_data.shape)
                                raise ValueError(f"Data dimension too large in condition {cond}: {cond_data.shape}")
                    except (KeyError, IndexError):
                        # Condition doesn't exist in data, skip validation
                        continue
                
                try:
                    for values in get_scores(
                            in_data, decoder, [list(range(subset.sum()))],
                            conds, [names[i]], on_gpu=on_cupy, shuffle=False,
                            which=wh, **window_kwargs):
                        key = decoder.current_job
                        _p("get_scores returned for key={} values_type={} shape={}", key, type(values), getattr(values, 'shape', None))
                        true_scores[key] = values
                        np.savez(true_name, **true_scores)
                except Exception as e:
                    _p("Exception while running get_scores on component {}: {}", i, e)
                    raise
    else:
        true_scores = dict(np.load(true_name + '.npz', allow_pickle=True))
        _p("Loaded existing true_scores from {}.npz with keys: {}", true_name, list(true_scores.keys())[:5])

    plots = {}
    for key, values in true_scores.items():
        if values is None:
            continue
        plots[key] = np.mean(values.T[np.eye(n_classes).astype(bool)].T, axis=2)
    fig, axs = plot_all_scores(plots, conds,
                               {n: i for n, i in zip(names, idxs)},
                               colors, "Word Decoding", ylims=(
            baseline-0.2, baseline + 0.6))

    decoder = Decoder({'heat': 0, 'hoot': 1, 'hot': 2, 'hut': 3},
                      5, 50, 1, 'train', model=decode_model)
    shuffle_name = 'shuffle_scores' + suffix

    if not os.path.exists(shuffle_name + '.npz'):
        with joblib.parallel_backend(backend):
            for i in range(n_components[0]):
                subset = np.nonzero(W[i]/W.mean() > 0.05)[0]
                # subset = np.nonzero(W[i] == np.max(W, 0))[0]

                in_data = zscores[:,:,[labels[0][s] for s in subset], ..., :aud_len]
                weights = model.construct_single_component(0, i).detach().numpy()[subset]
                weights_aud = np.nanmean(weights[None, ..., None, :aud_len], axis=2)
                weights_go = np.nanmean(weights[None, ..., None, aud_len:], axis=2)
                for c in ['ls', 'lm', 'jl']:
                    weighted_preserve_stats(in_data['aud_' + c], weights_aud)
                    weighted_preserve_stats(in_data['go_' + c], weights_go)
                for values in get_scores(in_data, decoder, [list(range(subset.sum()))], conds,
                                         [names[i]], on_gpu=on_cupy, shuffle=True,
                                         which=wh, **window_kwargs):
                    key = decoder.current_job
                    shuffle_scores[key] = values

                    np.savez(shuffle_name, **shuffle_scores)
    else:
        shuffle_scores = dict(np.load(shuffle_name + '.npz', allow_pickle=True))

        # shuffle_score['All-aud_ls-aud_lm'] = shuffle_score['Auditory-aud_ls-aud_lm']
        # shuffle_score['All-go_ls-go_lm'] = shuffle_score['Production-go_ls-go_lm']
        # shuffle_score['All-resp'] = shuffle_score['Production-resp']

    # Time Sliding decoding significance

    signif = {}
    pvals = {}
    for cond, score in true_scores.items():
        if cond.split('-')[0] not in names:
            continue
        true = np.mean(score.T[np.eye(n_classes).astype(bool)].T, axis=2)
        shuffle = np.mean(shuffle_scores[cond].T[np.eye(n_classes).astype(bool)].T,
                          axis=2)
        signif[cond], pvals[cond] = time_perm_cluster(#true.T,
            true.mean(axis=1, keepdims=True).T,
            shuffle.T, 0.05, n_perm=10000,
            stat_func=lambda x, y, axis: np.mean(x, axis=axis)
        )

    # Ray cluster processes are managed by the SLURM launcher; no cleanup here.

    # Plot significance
    for cond, ax in zip(conds, axs):
        if cond.split('-')[0] not in names:
            continue
        bars = []
        if isinstance(cond, list):
            cond = "-".join(cond)
        for i, idx in enumerate(idxs):
            name = "-".join([names[i], cond])
            if name.endswith('resp'):
                times = (-1, 1)
            else:
                times = (-0.5, 1.25)
            shuffle = np.mean(shuffle_scores[name].T[np.eye(n_classes).astype(bool)].T,
                              axis=2)
            # smooth the shuffle using a window
            window = np.lib.stride_tricks.sliding_window_view(shuffle, 20, 0)
            shuffle = np.mean(window, axis=-1)
            plot_dist_bound(shuffle, 'std', 'both', times, 0, ax=ax,
                            color=colors[i], alpha=0.3)
            bars.append(signif[name])
        plot_horizontal_bars(ax, bars, 0.02, 'below')

    for ax in fig.axes:
        ax.axhline(1 / n_classes, color='k', linestyle='--')
        ax.set_ylim(baseline-0.2, baseline + 0.6)
        ax.legend()
