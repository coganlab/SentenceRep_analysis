import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
cm = 1 / 2.54
LABEL_SIZE = 7
TICK_SIZE = 5

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
