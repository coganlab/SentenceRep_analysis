"""
Plot trial-averaged periodic-component spectrograms from SpecParam time-event models.

Usage (powershell):
  python analysis/check/check_chans_periodic.py
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

import mne.time_frequency
import numpy as np
import matplotlib.pyplot as plt
from ieeg.calc.scaling import rescale
from ieeg.io import get_data
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from specparam import SpectralTimeEventModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================
# CONFIGURATION SECTION (EDIT HERE)
# =============================

BIDS_ROOT_BOX = os.path.join(os.path.expanduser("~"), "Box", "CoganLab")
BIDS_ROOT_WORKSPACE = os.path.join(os.path.expanduser("~"), "workspace", "CoganLab")

DATASET_NAME = "SentenceRep"

DERIVATIVES_SPEC = os.path.join("derivatives", "spec", "super_log")
DERIVATIVES_SPECPARAM = os.path.join("derivatives", "specparam")

CONDITIONS = ["start", "resp", "aud_ls", "aud_lm", "aud_jl", "go_jl", "go_ls", "go_lm"]

PERIODIC_COMPONENT = "peak"
SPACE = "linear"

# =============================
# END CONFIGURATION SECTION
# =============================


def _read_tfr(path: str) -> Optional[mne.time_frequency.BaseTFR]:
    try:
        spec = mne.time_frequency.read_tfrs(path)
    except OSError:
        return None
    if isinstance(spec, list):
        spec = spec[0]
    if hasattr(spec, "average"):
        spec = spec.average(lambda x: np.nanmean(x, axis=0))
    return spec


def _load_event_identities(model_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(model_dir, "event_identities.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _load_time_event_model(model_dir: str) -> SpectralTimeEventModel:
    file_name = None
    for fname in os.listdir(model_dir):
        if fname.startswith("model_") and (fname.endswith(".json") or fname.endswith(".json.gz")):
            file_name = "model"
            break
    if file_name is None:
        for candidate in ("model", "model.json", "model.json.gz"):
            if os.path.exists(os.path.join(model_dir, candidate)):
                file_name = "model"
                break
    if file_name is None:
        for fname in os.listdir(model_dir):
            if fname.endswith(".json.gz"):
                base = fname[:-8]
            elif fname.endswith(".json"):
                base = fname[:-5]
            else:
                continue
            if base.endswith(tuple(str(d) for d in range(10))) and "_" in base:
                base = base.rsplit("_", 1)[0]
            file_name = base
            break
    if file_name is None:
        raise FileNotFoundError(f"No model file found in {model_dir}")

    load_event = None
    try:
        from specparam.io import load_event as load_event
    except Exception:
        try:
            from specparam import load_event as load_event
        except Exception:
            load_event = None

    if load_event is not None:
        try:
            return load_event(file_name, model_dir)
        except Exception:
            if file_name.endswith(".json"):
                return load_event(file_name[:-5], model_dir)
            if file_name.endswith(".json.gz"):
                return load_event(file_name[:-8], model_dir)
            return load_event(file_name, model_dir)

    model = SpectralTimeEventModel()
    try:
        model.load(file_name, model_dir)
    except Exception:
        model.load(os.path.join(model_dir, file_name))
    return model


def _get_model_freqs(model: SpectralTimeEventModel) -> np.ndarray:
    freqs = getattr(model, "freqs", None)
    if freqs is None:
        freqs = getattr(model, "_freqs", None)
    if freqs is None:
        data = getattr(model, "data", None)
        freqs = getattr(data, "freqs", None) if data is not None else None
        if freqs is None and data is not None and hasattr(data, "_regenerate_freqs"):
            if getattr(data, "freq_range", None) is not None and getattr(data, "freq_res", None) is not None:
                data._regenerate_freqs()
                freqs = getattr(data, "freqs", None)
    if freqs is None:
        raise AttributeError("Model frequencies not found.")
    return freqs


def _infer_event_window_counts(model: SpectralTimeEventModel) -> tuple[int, int]:
    event_group = getattr(model.results, "event_group_results", None)
    if event_group:
        n_events = len(event_group)
        lengths = [len(entry) for entry in event_group if entry is not None]
        n_windows = min(lengths) if lengths else 0
        return n_events, n_windows
    event_time = getattr(model.results, "event_time_results", None)
    if event_time:
        sample = next(iter(event_time.values()))
        if hasattr(sample, "shape") and len(sample.shape) == 2:
            return int(sample.shape[0]), int(sample.shape[1])
    return 0, 0


def _compute_periodic_component(
    model: SpectralTimeEventModel, space: str = SPACE
) -> np.ndarray:
    n_events, n_times = _infer_event_window_counts(model)
    if n_events == 0 or n_times == 0:
        raise RuntimeError(
            "Model results are missing; re-run fitting with save_results=True or save_data=True."
        )
    n_freqs = _get_model_freqs(model).shape[0]
    periodic = np.full((n_events, n_freqs, n_times), np.nan, dtype=float)
    logger.info("periodic dims: n_events=%d n_times=%d n_freqs=%d", n_events, n_times, n_freqs)
    get_models = getattr(model, "get_models", None)
    all_models = None
    if callable(get_models):
        try:
            all_models = get_models(regenerate=True)
        except TypeError:
            try:
                all_models = get_models()
            except Exception as exc:
                logger.info("get_models() failed, falling back to get_model: %s", exc)
                all_models = None
        except Exception as exc:
            logger.info("get_models() failed, falling back to get_model: %s", exc)
            all_models = None

    for event_idx in range(n_events):
        if event_idx == 0 or (event_idx + 1) % 5 == 0 or event_idx == n_events - 1:
            logger.info("periodic progress: event %d/%d", event_idx + 1, n_events)
        if all_models is not None:
            try:
                if len(all_models) == n_events and hasattr(all_models[event_idx], "__len__"):
                    event_models = all_models[event_idx]
                else:
                    event_models = all_models
            except Exception:
                event_models = None
        else:
            event_models = None

        if event_models is None:
            event_models = []
            for time_idx in range(n_times):
                if event_idx == 0 and (
                    time_idx == 0 or (time_idx + 1) % 10 == 0 or time_idx == n_times - 1
                ):
                    logger.info("periodic progress: event %d time %d/%d", event_idx + 1, time_idx + 1, n_times)
                event_models.append(model.get_model(event_idx, time_idx, regenerate=True))

        for time_idx, spec_model in enumerate(event_models):
            if event_idx == 0 and (
                time_idx == 0 or (time_idx + 1) % 10 == 0 or time_idx == n_times - 1
            ):
                logger.info("periodic progress: event %d time %d/%d", event_idx + 1, time_idx + 1, n_times)
            if spec_model is None:
                continue
            components = getattr(spec_model.results, "model", None)
            if components is None or not hasattr(components, "get_component"):
                raise RuntimeError("Model components are unavailable for periodic extraction.")
            periodic[event_idx, :, time_idx] = components.get_component(
                PERIODIC_COMPONENT, space=space
            )
    return periodic


def _parse_event_index(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _group_event_indices(
    identities: List[Dict[str, str]],
    conditions: List[str],
    n_events: int,
) -> Dict[str, List[int]]:
    grouped = {cond: [] for cond in conditions}
    for row_idx, row in enumerate(identities):
        condition = row.get("condition")
        if condition not in grouped:
            continue
        candidates: List[int] = []
        for key in ("event_index", "combined_event_index"):
            parsed = _parse_event_index(row.get(key))
            if parsed is not None:
                candidates.append(parsed)
        if not candidates:
            candidates.append(row_idx)
        idx = next((cand for cand in candidates if 0 <= cand < n_events), None)
        if idx is None and 0 <= row_idx < n_events:
            idx = row_idx
        if idx is None:
            logger.warning(
                "Skipping identity with out-of-range index for condition %s: %s",
                condition,
                row,
            )
            continue
        grouped[condition].append(idx)
    for cond in grouped:
        grouped[cond] = sorted(set(grouped[cond]))
    return grouped


def main() -> None:
    logger.info("starting check_chans_periodic")
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        lab_root = BIDS_ROOT_WORKSPACE
    else:
        lab_root = BIDS_ROOT_BOX

    layout = get_data(DATASET_NAME, root=lab_root)
    subjects = layout.get(return_type="id", target="subject")
    logger.info("loaded layout: n_subjects=%d", len(subjects))

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        subject_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        subjects = [subjects[subject_idx]] if subject_idx < len(subjects) else []
        logger.info("slurm subject index=%d -> n_subjects=%d", subject_idx, len(subjects))

    fig_path = os.path.join(layout.root, DERIVATIVES_SPECPARAM, "periodic_figs")
    os.makedirs(fig_path, exist_ok=True)
    logger.info("output dir: %s", fig_path)

    for subj in subjects:
        logger.info("subject start: %s", subj)
        subject_model_dir = os.path.join(layout.root, DERIVATIVES_SPECPARAM, subj)
        if not os.path.exists(subject_model_dir):
            logger.warning("Missing specparam output for %s", subj)
            continue

        channel_dirs = {
            name.replace("_time_event_model", ""): os.path.join(subject_model_dir, name)
            for name in os.listdir(subject_model_dir)
            if name.endswith("_time_event_model")
        }
        logger.info("time-event model dirs: %d", len(channel_dirs))
        if not channel_dirs:
            logger.warning("No time-event models found for %s", subj)
            continue

        channel_periodic: Dict[str, Dict[str, np.ndarray]] = {}
        for ch_name, model_dir in channel_dirs.items():
            logger.info("loading model: %s", ch_name)
            try:
                model = _load_time_event_model(model_dir)
            except Exception as exc:
                logger.warning("Failed to load model for %s: %s", ch_name, exc)
                continue
            identities = _load_event_identities(model_dir)
            if not identities:
                logger.warning("Missing event identities for %s", ch_name)
                continue

            logger.info("computing periodic component: %s", ch_name)
            periodic = _compute_periodic_component(model, space=SPACE)
            n_events = periodic.shape[0]
            channel_periodic[ch_name] = {
                "periodic": periodic,
                "freqs": _get_model_freqs(model),
                "groups": _group_event_indices(identities, CONDITIONS, n_events),
            }

        if not channel_periodic:
            logger.warning("No periodic data loaded for %s", subj)
            continue

        base = None
        for cond in CONDITIONS:
            logger.info("condition start: %s %s", subj, cond)
            tfr_path = os.path.join(
                layout.root, DERIVATIVES_SPEC, subj, f"{cond}-tfr.h5"
            )
            spec = _read_tfr(tfr_path)
            if spec is None:
                logger.warning("Skipping %s %s: missing TFR", subj, cond)
                continue

            ref = next(iter(channel_periodic.values()))
            freqs = ref["freqs"]
            times = spec.times
            n_times = ref["periodic"].shape[-1]
            if times.shape[0] != n_times:
                n_times = min(times.shape[0], n_times)
                times = times[:n_times]
            data = np.full(
                (len(spec.ch_names), len(freqs), n_times), np.nan, dtype=float
            )
            nave = 0

            for ch_idx, ch_name in enumerate(spec.ch_names):
                if ch_name not in channel_periodic:
                    continue
                ch_entry = channel_periodic[ch_name]
                event_indices = ch_entry["groups"].get(cond, [])
                if event_indices:
                    max_events = ch_entry["periodic"].shape[0]
                    filtered = [idx for idx in event_indices if 0 <= idx < max_events]
                    if len(filtered) != len(event_indices):
                        logger.warning(
                            "Dropping %d out-of-range events for %s %s %s",
                            len(event_indices) - len(filtered),
                            subj,
                            ch_name,
                            cond,
                        )
                    event_indices = filtered
                if not event_indices:
                    continue
                spectra = ch_entry["periodic"][event_indices, :, :n_times]
                data[ch_idx] = np.nanmean(spectra, axis=0)
                nave = max(nave, len(event_indices))

            periodic_tfr = mne.time_frequency.AverageTFR(
                info=spec.info.copy(),
                data=data,
                times=times,
                freqs=freqs,
                nave=max(nave, 1),
                comment=cond,
            )

            if cond == "start":
                logger.info("setting baseline from start condition")
                base = periodic_tfr.copy().crop(tmin=-0.5, tmax=0)
            if base is not None:
                logger.info("rescaling %s %s", subj, cond)
                rescale(periodic_tfr, base, mode="ratio", copy=False)

            logger.info("building channel grid figures: %s %s", subj, cond)
            figs = chan_grid(
                periodic_tfr,
                size=(20, 10),
                vlim=(0.7, 1.4),
                cmap=parula_map,
                show=False,
            )
            logger.info("channel grid done: nfigs=%d", len(figs))
            for i, fig in enumerate(figs):
                logger.info("saving figure %d/%d for %s %s", i + 1, len(figs), subj, cond)
                fig.savefig(
                    os.path.join(fig_path, f"{subj}_{cond}_{i + 1}.jpg"),
                    bbox_inches="tight",
                )
                plt.close(fig)
            logger.info("condition done: %s %s", subj, cond)
        logger.info("subject done: %s", subj)
    logger.info("check_chans_periodic complete")


if __name__ == "__main__":
    main()
