"""Supplementary Figure: ROI breakdown of AUD / SM / PROD functional groups.

For each of the three canonical functional groupings (AUD, SM, PROD), counts
how many electrodes fall into each Brainnetome gyrus-level ROI and draws a
stacked bar chart per group.  Mirrors the per-component ROI block at
``analysis/decomposition/slice_freq.py:840-909`` but switches the indexing
from sliceTCA membership to the AUD/SM/PROD sets returned by ``group_elecs``.
"""
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from analysis.figures.config import (
    cm, setup_figure, LABEL_SIZE, TICK_SIZE,
    LAYOUT, finalize_figure, add_panel_label,
)
from analysis.grouping import group_elecs
from ieeg.arrays.label import LabeledArray
from ieeg.viz.mri import gen_labels, subject_to_info, Atlas

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------
layout = LAYOUT
folder = "stats_freq_hilbert"
derivatives = os.path.join(layout.root, "derivatives", folder, "combined")

# Significance mask -> AUD / SM / PROD sets via the same code path as fig 2
sigs = LabeledArray.fromfile(os.path.join(derivatives, "mask"))
AUD, SM, PROD, sig_chans, delay = group_elecs(
    sigs, sigs.labels[1], sigs.labels[0]
)
all_chans = sigs.labels[1]

# ROI vocabulary copied verbatim from slice_freq.py:844-847
ROIS = [
    'IFG', 'Tha', 'PoG', 'Amyg', 'PhG', 'MVOcC', 'ITG', 'PrG', 'PCL',
    'IPL', 'MFG', 'CG', 'Pcun', 'BG', 'INS', 'FuG', 'LOcC', 'STG',
    'OrG', 'MTG', 'pSTS', 'Hipp', 'SFG', 'SPL',
]
atlas = Atlas()

GROUPS = [
    ("Auditory",      AUD,  "green"),
    ("Sensory-Motor", SM,   "red"),
    ("Production",    PROD, "blue"),
]


def _count_rois(elec_indices: set[int]) -> dict[str, int]:
    """Map a set of electrode indices into their ROI counts."""
    target = {all_chans[i] for i in elec_indices}
    counts = {r: 0 for r in ROIS}
    for subj in layout.get_subjects():
        subj_old = f"D{int(subj[1:])}"
        try:
            info = subject_to_info(subj_old)
        except Exception:
            continue
        ch_labels = gen_labels(info, subj_old, atlas='.BN_atlas')
        for ch, val in ch_labels.items():
            full = f"{subj}-{ch}"
            if full not in target:
                continue
            base = val.split("_")[0]
            try:
                gyrus = atlas[base].gyrus
            except KeyError:
                if base == 'TE1.0/TE1.2':
                    gyrus = 'STG'
                else:
                    continue
            if gyrus in counts:
                counts[gyrus] += 1
    return counts


roi_counts = [(name, color, _count_rois(idx)) for name, idx, color in GROUPS]

# Keep only the top 10 ROIs by total count across all groups
TOP_N = 10
totals = {r: sum(c[r] for _, _, c in roi_counts) for r in ROIS}
keep_rois = sorted([r for r in ROIS if totals[r] > 0],
                   key=lambda r: -totals[r])[:TOP_N]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = setup_figure(figsize=(18 * cm, 12 * cm))
gs = gridspec.GridSpec(len(GROUPS), 1, figure=fig, hspace=0.35)

ymax = max(c[r] for _, _, c in roi_counts for r in keep_rois)
for i, (name, color, counts) in enumerate(roi_counts):
    ax = fig.add_subplot(gs[i, 0])
    bars = ax.bar(keep_rois, [counts[r] for r in keep_rois], color=color)
    ax.set_ylabel(f"{name}\n(n={sum(counts[r] for r in keep_rois)})",
                  fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_ylim(0, ymax * 1.05)
    if i == len(GROUPS) - 1:
        ax.set_xticks(range(len(keep_rois)))
        ax.set_xticklabels(keep_rois, rotation=45, ha='right',
                           fontsize=TICK_SIZE)
    else:
        ax.set_xticks([])
    add_panel_label(ax, "abc"[i], x=-0.06, y=1.05)
    # Print summary to stdout for the user to paste into the paper
    print(f"\n{name} (n={sum(counts[r] for r in keep_rois)})")
    for r in keep_rois:
        if counts[r]:
            print(f"  {r:<6} {counts[r]:>4}")

ax.set_xlabel("Brainnetome gyrus-level ROI", fontsize=LABEL_SIZE)

finalize_figure(fig, "figure_S_roi_groups")
