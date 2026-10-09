"""Plot style shared by the notebooks; importing it applies the matplotlib settings.

from plot_style import BLUE, ORANGE, SEQUENTIAL, no_grid
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
PANEL = '#f0efec'  # background behind drawn pixels
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]  # fixed order

SEQUENTIAL = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#2a78d6', '#104281', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list('blue_red', ['#104281', '#2a78d6', '#cde2fb', '#f0efec', '#fbd3c9', '#e34948', '#a51d1b'])

plt.rcParams.update(
    {
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'figure.dpi': 110,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.edgecolor': '#c3c2b7',
        'axes.labelcolor': INK_2,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'text.color': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.prop_cycle': plt.cycler(color=SERIES),
        'legend.frameon': False,
    }
)


def no_grid(ax):
    """Bare axes for images: no grid, ticks or spines."""
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
