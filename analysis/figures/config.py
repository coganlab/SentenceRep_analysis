import os

import matplotlib.pyplot as plt
from ieeg.io import get_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
cm = 1 / 2.54
LABEL_SIZE = 7
TICK_SIZE = 5

# SliceTCA component names and colours (shared across all figures)
COMP_NAMES = ["Auditory", "WM", "Motor", "Visual"]
COMP_COLORS_LIST = ["orange", "#4B0082", "c", "y"]
COMP_COLORS = dict(zip(COMP_NAMES, COMP_COLORS_LIST))

# ---------------------------------------------------------------------------
# Common paths
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
if "SLURM_ARRAY_TASK_ID" in os.environ:
    LAB_ROOT = os.path.join(HOME, "workspace", "CoganLab")
else:
    LAB_ROOT = os.path.join(HOME, "Box", "CoganLab")

LAYOUT = get_data("SentenceRep", root=LAB_ROOT)

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(FIGURES_DIR)
DECODING_DIR = os.path.join(ANALYSIS_DIR, "decoding")
DECOMPOSITION_DIR = os.path.join(ANALYSIS_DIR, "decomposition")

SM_PKL = os.path.join(DECODING_DIR, "SM_chns.pkl")
SM_MODEL = os.path.join(DECOMPOSITION_DIR, "model_SM7_freq.pt")

EXCLUDE = [
    "D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
    "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
    "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
    "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
    "D0027-LPIF10", "D0026-RPG20", "D0026-RPG21", "D0026-RPG28",
    "D0026-RPG29", "D0026-RPG36", "D0007-RFG44",
]

# ---------------------------------------------------------------------------
# Plotting settings
# ---------------------------------------------------------------------------
def setup_figure(figsize=(18 * cm, 12 * cm)):
    """Initialize a figure with consistent size and style."""
    fig = plt.figure(figsize=figsize)
    # Global rcParams can also be set here if needed
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    return fig

def despine(ax):
    """Remove top and right axis lines (spines)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# GridSpec defaults
# Using small hspace and wspace as requested
GS_KWARGS = {'hspace': 0.15, 'wspace': 0.05}
