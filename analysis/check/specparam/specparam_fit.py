"""
Fit SpecParam time-resolved spectral models for each channel of a subject.

Usage (powershell):
  python analysis/check/specparam_fit.py

Behaviour:
- Finds TFRs under derivatives/spec/super/{subj}/{cond}-tfr.h5 using ieeg.io.get_data() to locate the dataset root.
- For each channel, combines all conditions into a single events array and fits a single SpectralTimeEventModel.
- Saves one time-event model per channel along with an event-identity mapping (condition + event index).

Notes:
- This script attempts to import SpectralTimeEventModel from specparam.
- Default behaviour operates on trial/event TFRs (EpochsTFR); if only averaged data are present, each
  condition contributes a single event.

"""
from __future__ import annotations

import csv
import logging
import os
from typing import Optional

import numpy as np
import mne.time_frequency
from ieeg.io import get_data
from ieeg.timefreq.utils import resample_tfr_freqs

# Try to import SpectralTimeEventModel from a few likely locations in specparam
from specparam import SpectralTimeEventModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================
# CONFIGURATION SECTION (EDIT HERE)
# =============================

# Path to BIDS root (Box or workspace)
BIDS_ROOT_BOX = os.path.join(os.path.expanduser("~"), "Box", "CoganLab")
BIDS_ROOT_WORKSPACE = os.path.join(os.path.expanduser("~"), "workspace", "CoganLab")

# Dataset name for ieeg.io.get_data()
DATASET_NAME = "SentenceRep"

# Output directory (relative to BIDS root)
DERIVATIVES_SPEC = os.path.join("derivatives", "spec", "super_log")
DERIVATIVES_SPECPARAM = os.path.join("derivatives", "specparam")

# Frequency range for fitting (min_freq, max_freq)
FREQ_RANGE = (2, 150)

# Minimum number of frequency points required for fitting
MIN_FREQ_POINTS = 10

# Window size and step size (seconds)
WINDOW_SIZE = .1  # 500 ms
STEP_SIZE = 0.25   # 250 ms

MAX_PEAKS = 10  # Maximum number of peaks to fit
MIN_PEAK_HEIGHT = 0.5  # Maximum peak height

# Conditions to process (edit as needed)
CONDITIONS = {
    'start': 'Start',
    'resp': 'Word/Response/LS',
    'aud_ls': 'Word/Audio/LS',
    'aud_lm': 'Word/Audio/LM',
    'aud_jl': 'Word/Audio/JL',
    'go_ls': 'Word/Go/LS',
    'go_lm': 'Word/Go/LM',
    'go_jl': 'Word/Go/JL'
}

# =============================
# END CONFIGURATION SECTION
# =============================

# Update all references in the script to use these variables instead of hard-coded values.
# For example, replace freq_range=(1, 50) with freq_range=FREQ_RANGE, window_size=WINDOW_SIZE, etc.



def load_tfr_data(layout, subject: str, condition: str) -> Optional[mne.time_frequency.BaseTFR]:
    """
    Load TFR data for a given subject and condition, using the same robust pattern as check_chans.py.
    """
    tfr_dir = os.path.join(layout.root, DERIVATIVES_SPEC, subject)
    tfr_file = os.path.join(tfr_dir, f'{condition}-tfr.h5')
    if not os.path.exists(tfr_file):
        logger.warning(f"TFR file not found: {tfr_file}")
        return None
    try:
        spec = mne.time_frequency.read_tfrs(tfr_file)
        return spec
    except OSError:
        logger.warning(f"Skipping {subject} {condition}")
        return None
    except Exception as e:
        logger.error(f"Error loading TFR for {subject} {condition}: {e}")
        return None


def average_tfr_epochs(tfr: mne.time_frequency.BaseTFR) -> mne.time_frequency.BaseTFR:
    """
    Average TFR across epochs if it's an EpochsTFR.

    Parameters
    ----------
    tfr : mne.time_frequency.BaseTFR
        Input TFR (may be EpochsTFR or AverageTFR)

    Returns
    -------
    tfr : mne.time_frequency.BaseTFR
        Averaged TFR
    """
    if hasattr(tfr, 'average'):
        # This is an EpochsTFR
        tfr = tfr.average()
    return tfr


def window_tfr_data(
    tfr: mne.time_frequency.BaseTFR,
    window_size: float,
    step_size: Optional[float] = None
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Window the time axis of TFR data and compute mean power spectrum per window.

    Parameters
    ----------
    tfr : mne.time_frequency.BaseTFR
        Input TFR data
    window_size : float
        Window size in seconds
    step_size : float, optional
        Step size in seconds. If None, uses window_size (non-overlapping windows)

    Returns
    -------
    windowed_spectra : np.ndarray
        Shape: (n_windows, n_channels, n_freqs)
        Mean power spectrum for each window and channel
    window_centers : np.ndarray
        Shape: (n_windows,)
        Time of window centers in seconds
    freqs : np.ndarray
        Shape: (n_freqs,)
        Frequency vector
    """
    if step_size is None:
        step_size = window_size

    # Get TFR data: shape (n_channels, n_freqs, n_times) or (n_freqs, n_times)
    data = tfr.data
    if data.ndim == 2:
        # Add channel dimension
        data = data[np.newaxis, :, :]

    n_channels, n_freqs, n_times = data.shape
    times = tfr.times
    freqs = tfr.freqs

    # Calculate window boundaries
    window_size_samples = int(window_size * tfr.info['sfreq'])
    step_size_samples = int(step_size * tfr.info['sfreq'])

    windows: list[np.ndarray] = []
    window_centers: list[float] = []
    window_idx = 0

    while window_idx * step_size_samples + window_size_samples <= n_times:
        start_idx = window_idx * step_size_samples
        end_idx = start_idx + window_size_samples

        # Extract window and average across time
        window_data = data[:, :, start_idx:end_idx]
        window_mean = np.mean(window_data, axis=2)  # Shape: (n_channels, n_freqs)
        windows.append(window_mean)

        # Calculate window center time
        center_time = times[start_idx + window_size_samples // 2]
        window_centers.append(center_time)

        window_idx += 1

    if len(windows) == 0:
        logger.warning("No windows created - data may be too short")
        return None, None, freqs

    windowed_spectra: np.ndarray = np.array(windows)  # Shape: (n_windows, n_channels, n_freqs)
    window_centers: np.ndarray = np.array(window_centers)

    return windowed_spectra, window_centers, freqs


def _save_event_identities(event_identities: list[dict], output_dir: str) -> None:
    if not event_identities:
        return
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "event_index",
        "combined_event_index",
        "subject",
        "channel",
        "condition",
        "condition_event_index",
        "metadata_index",
    ]
    out_path = os.path.join(output_dir, "event_identities.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for identity in event_identities:
            writer.writerow({key: identity.get(key, "") for key in fieldnames})


def fit_time_event_model(
    spectrograms: np.ndarray,
    freqs: np.ndarray,
    subject: str,
    chan: str,
    event_identities: list[dict],
    save_dir: str,
    **kwargs,
) -> None:
    if spectrograms.ndim != 3:
        raise ValueError(f"Expected spectrograms with 3 dims, got {spectrograms.shape}")
    lower = 2 * (freqs[1] - freqs[0])
    peak_width_limits = (lower, lower * 20)
    kwargs2 = dict(
        peak_width_limits=peak_width_limits,
        aperiodic_mode='fixed',
        periodic_mode='gaussian',
        max_n_peaks=MAX_PEAKS,
        peak_threshold=MIN_PEAK_HEIGHT,
    )

    model = SpectralTimeEventModel(**kwargs2)
    model.fit(freqs, spectrograms, freq_range=FREQ_RANGE, **kwargs)

    model_dir = os.path.join(save_dir, f"{chan}_time_event_model")
    os.makedirs(model_dir, exist_ok=True)
    model.save("model", model_dir, False, True, True, True)
    model.save_report("report", model_dir)
    _save_event_identities(event_identities, model_dir)


def main():
    """Main execution function."""
    HOME = os.path.expanduser("~")

    # Use config section variables
    import sys
    # Use config for BIDS root
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        LAB_root = BIDS_ROOT_WORKSPACE
    else:
        LAB_root = BIDS_ROOT_BOX
    layout = get_data(DATASET_NAME, root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        subject_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
        subjects_to_process = [subjects[subject_idx]] if subject_idx < len(subjects) else []
    else:
        subjects_to_process = subjects

    # Output directory
    output_dir = os.path.join(layout.root, DERIVATIVES_SPECPARAM)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CSV file for summary results
    csv_file = os.path.join(output_dir, 'specparam_summary.csv')
    csv_exists = os.path.exists(csv_file)

    # Process each subject
    for subject in subjects_to_process:
        logger.info(f"Processing subject {subject}")
        subject_output_dir = os.path.join(output_dir, subject)
        if not os.path.exists(subject_output_dir):
            os.makedirs(subject_output_dir)

        condition_entries: list[dict] = []
        ref_ch_names: Optional[list[str]] = None
        min_n_times: Optional[int] = None
        new_freqs = np.linspace(FREQ_RANGE[0], FREQ_RANGE[1], 200)

        # Load all conditions once so events can be combined per channel.
        for cond_name in CONDITIONS:
            logger.info(f"  Loading condition: {cond_name}")
            tfr = load_tfr_data(layout, subject, cond_name)
            if tfr is None:
                logger.warning(f"Skipping {subject} {cond_name} - TFR not found")
                continue
            resample_tfr_freqs(tfr, new_freqs, tfr.freqs)
            data = tfr.get_data()
            if data.ndim not in (3, 4):
                logger.warning(f"Unexpected TFR shape for {subject} {cond_name}: {data.shape}")
                continue
            n_times = data.shape[-1]
            min_n_times = n_times if min_n_times is None else min(min_n_times, n_times)
            ch_names = list(tfr.ch_names)
            if ref_ch_names is None:
                ref_ch_names = ch_names
            condition_entries.append(
                {
                    "name": cond_name,
                    "data": data,
                    "ch_names": ch_names,
                    "ch_index": {name: idx for idx, name in enumerate(ch_names)},
                    "metadata": getattr(tfr, "metadata", None),
                }
            )

        if not condition_entries or ref_ch_names is None or min_n_times is None:
            logger.warning(f"No usable conditions found for subject {subject}")
            continue

        freqs = new_freqs

        for ch_name in ref_ch_names:
            # check
            if os.path.exists(
                    os.path.join(
                        subject_output_dir, f"{ch_name}_time_event_model",
                        "model_000.json"
                    )
            ):
                logger.info(
                    f"Skipping {subject} {ch_name} - model already exists"
                )
                continue
            combined_spectrograms: list[np.ndarray] = []
            combined_identities: list[dict] = []
            for cond_entry in condition_entries:
                if ch_name not in cond_entry["ch_index"]:
                    logger.warning(
                        f"Channel {ch_name} missing in condition {cond_entry['name']} for {subject}"
                    )
                    continue
                ch_idx = cond_entry["ch_index"][ch_name]
                data = cond_entry["data"]
                if data.ndim == 4:
                    ch_data = data[:, ch_idx, :, :min_n_times]
                else:
                    ch_data = data[ch_idx, :, :min_n_times][np.newaxis, :, :]
                n_events = ch_data.shape[0]
                metadata = cond_entry.get("metadata")
                for event_idx in range(n_events):
                    combined_index = len(combined_identities)
                    identity = {
                        "combined_event_index": combined_index,
                        "subject": subject,
                        "channel": ch_name,
                        "condition": cond_entry["name"],
                        "condition_event_index": event_idx,
                    }
                    if metadata is not None and hasattr(metadata, "index") and len(metadata) == n_events:
                        identity["metadata_index"] = str(metadata.index[event_idx])
                    combined_identities.append(identity)
                combined_spectrograms.append(ch_data)

            if not combined_spectrograms:
                logger.warning(f"No data for channel {ch_name} in subject {subject}")
                continue

            spectrograms = np.concatenate(combined_spectrograms, axis=0)
            valid_events = ~np.isnan(spectrograms[:, :, 0]).any(axis=1)
            if not np.all(valid_events):
                logger.info(
                    f"Dropping {np.sum(~valid_events)} invalid events for {subject} {ch_name}"
                )
            spectrograms = spectrograms[valid_events]
            event_identities = [
                ident for ident, ok in zip(combined_identities, valid_events) if ok
            ]
            for event_index, identity in enumerate(event_identities):
                identity["event_index"] = event_index

            if spectrograms.size == 0 or not event_identities:
                logger.warning(f"No valid events for {subject} {ch_name}")
                continue

            fit_time_event_model(
                spectrograms,
                freqs,
                subject,
                ch_name,
                event_identities,
                subject_output_dir,
                n_jobs=10,
                progress='tqdm'
            )

    #         for i, ch in tqdm(enumerate(tfr.ch_names)):
    #             valid_events = ~np.isnan(windowed_spectras[:, i, :, 0]).any(axis=1)
    #             windowed_spectra = windowed_spectras[valid_events, i]
    #             # Fit SpecParam models
    #             model, aperiodic_param, peak_param, r_squared, error = fit_specparam_models(
    #                 windowed_spectra, freqs, n_jobs=-1
    #             )
    #
    #             # Create result object
    #             result = SpecParamResult(
    #                 subject=subject,
    #                 channel=ch,
    #                 condition=cond_name,
    #                 n_windows=windowed_spectra.shape[-1],
    #                 window_size=WINDOW_SIZE,
    #                 models=model,
    #                 aperiodic_params=np.array(aperiodic_param),
    #                 peak_params=np.array(peak_param),
    #                 r_squared=np.array(r_squared),
    #                 error=np.array(error),
    #                 freqs=freqs
    #             )
    #             # Save models to disk
    #             models_file = os.path.join(subject_output_dir, f'{cond_name}_{ch}_models.joblib')
    #             joblib.dump(result, models_file)
    #             logger.info(f"    Saved models to {models_file}")
    #             results.append(result)
    #         # Add to CSV results
    #         for result in results:
    #             results_for_csv.append(result.__dict__)
    #     # Write CSV
    #     if results_for_csv:
    #         with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
    #             writer = csv.DictWriter(
    #                 f,
    #                 fieldnames=[
    #                     'subject', 'channel', 'condition', 'n_windows',
    #                     'window_size', 'models', 'aperiodic_params',
    #                     'peak_params', 'r_squared', 'error', 'freqs'
    #                 ]
    #             )
    #             if not csv_exists:
    #                 writer.writeheader()
    #             writer.writerows(results_for_csv)
    #         csv_exists = True
    #         logger.info(f"Updated CSV: {csv_file}")
    # logger.info("SpecParam fitting complete!")

if __name__ == '__main__':
    main()
