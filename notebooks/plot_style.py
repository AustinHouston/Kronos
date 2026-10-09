"""Plot style shared by the notebooks; importing it applies the matplotlib settings.

The base is SciencePlots' 'science' style (serif STIX text, inward ticks on all four sides,
minor ticks, thin frame) without LaTeX, so labels may use any unicode (µ, ×, −, σ, …).
Sizes are tuned for the wide multi-panel figures of the notebooks, and colors are ours.

from plot_style import BLUE, ORANGE, SEQUENTIAL, no_grid
"""

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' styles)
from matplotlib.colors import LinearSegmentedColormap

INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#ffffff'
PANEL = '#f0efec'  # background behind drawn pixels
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]  # fixed order

SEQUENTIAL = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#2a78d6', '#104281', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list('blue_red', ['#104281', '#2a78d6', '#cde2fb', '#f0efec', '#fbd3c9', '#e34948', '#a51d1b'])

plt.style.use(['science', 'no-latex'])
plt.rcParams.update(
    {
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'figure.dpi': 110,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'mathtext.fontset': 'stix',  # math in the same font as the text
        'axes.titlelocation': 'left',
        'axes.titlepad': 7,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'axes.linewidth': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'lines.linewidth': 1.5,
        'axes.grid': False,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.prop_cycle': plt.cycler(color=SERIES),
        'legend.frameon': False,
        'legend.handlelength': 1.6,
    }
)


def no_grid(ax):
    """Bare axes for images: no grid, ticks or spines."""
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
