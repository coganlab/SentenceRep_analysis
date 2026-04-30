import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from ieeg.io import get_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
cm = 1 / 2.54
LABEL_SIZE = 7
TICK_SIZE = 5
DPI = 300

# ---------------------------------------------------------------------------
# Global rcParams — Nature-compliant defaults
# ---------------------------------------------------------------------------
# Vector text in SVG (no rasterised glyphs)
mpl.rcParams['svg.fonttype'] = 'none'

# Font: Arial, 5-7 pt
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.labelsize'] = LABEL_SIZE
mpl.rcParams['xtick.labelsize'] = TICK_SIZE
mpl.rcParams['ytick.labelsize'] = TICK_SIZE
mpl.rcParams['legend.fontsize'] = TICK_SIZE

# Spines
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Ticks
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['ytick.major.width'] = 0.75
mpl.rcParams['xtick.minor.width'] = 0.75
mpl.rcParams['ytick.minor.width'] = 0.75
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.minor.size'] = 2

# Lines
mpl.rcParams['lines.linewidth'] = 0.75

# SliceTCA component names and colours (shared across all figures)
COMP_NAMES = ["Auditory", "WM", "Motor", "Visual"]
COMP_COLORS_LIST = ["orange", "#4B0082", "c", "y"]
COMP_COLORS = dict(zip(COMP_NAMES, COMP_COLORS_LIST))

# ---------------------------------------------------------------------------
# Event names and x-axis labels for time-aligned plots
# (single source of truth so capitalisation/spelling stay consistent)
# ---------------------------------------------------------------------------
EVENT_STIMULUS = "Stimulus"
EVENT_GO = "Go Cue"
EVENT_RESPONSE = "Response"

XLABEL_STIMULUS = f"Time from {EVENT_STIMULUS} (s)"
XLABEL_GO = f"Time from {EVENT_GO} (s)"
XLABEL_RESPONSE = f"Time from {EVENT_RESPONSE} (s)"

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
SM_MODEL = os.path.join(DECOMPOSITION_DIR, "model_SM2_freq.pt")

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
    return plt.figure(figsize=figsize)

def despine(ax):
    """Remove top and right axis lines (spines)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# GridSpec defaults
# Using small hspace and wspace as requested
GS_KWARGS = {'hspace': 0.15, 'wspace': 0.05}

# Standard label for colorbars that show power relative to baseline.
# Centralised here so all figures use the same capitalization and wording.
POWER_RATIO_LABEL = "Baseline Power Ratio"


def figure_out_path(name: str, out_dir: str | None = None) -> str:
    """Return the full path for a figure basename inside the figures dir.

    If out_dir is provided it is used, otherwise `FIGURES_DIR` is used.
    """
    out_dir = out_dir or FIGURES_DIR
    return os.path.join(out_dir, name)


def save_figure(fig, basename: str, out_dir: str | None = None,
                dpi: int = DPI, exts=("svg", "png")) -> None:
    """Save a figure to the figures directory in multiple formats.

    This centralises bbox/dpi settings so all figures are saved consistently.
    """
    out_dir = out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    for ext in exts:
        fig.savefig(os.path.join(out_dir, f"{basename}.{ext}"),
                    bbox_inches="tight", dpi=dpi)


def finalize_figure(fig, basename: str, out_dir: str | None = None,
                    dpi: int = DPI, exts=("svg", "png"), show: bool = True) -> None:
    """Save the figure (multiple formats) and optionally show it.
    """
    save_figure(fig, basename, out_dir=out_dir, dpi=dpi, exts=exts)
    if show:
        plt.show()


def add_panel_label(ax, letter: str, x: float = -0.05, y: float = 1.02,
                    fontsize: int | None = None, weight: str = "bold") -> None:
    """Add a standardized panel letter (e.g. 'a', 'b') to `ax`.

    Defaults match existing figures (uses `LABEL_SIZE + 2` when fontsize is None).
    """
    if fontsize is None:
        fontsize = LABEL_SIZE + 2
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight, va="bottom")


