""" This module contains dictionaries that can be used to set a
matplotlib plotting style.
It is mostly here to allow a consistent plotting style in tutorials,
but can be used to prepare any matplotlib figure.

Using a matplotlib version > 1.4 you can do::

    >>> import matplotlib.pyplot as pl
    >>> from gala.mpl_style import mpl_style
    >>> pl.style.use(mpl_style) # doctest: +SKIP

for older versions of matplotlib the following works::

    >>> import matplotlib as mpl
    >>> from gala.mpl_style import mpl_style
    >>> mpl.rcParams.update(mpl_style) # doctest: +SKIP

"""

mpl_style = {

    # Lines
    'lines.linewidth': 1.7,
    'lines.antialiased': True,
    'lines.marker': '.',
    'lines.markersize': 5.,

    # Patches
    'patch.linewidth': 1.0,
    'patch.facecolor': '#348ABD',
    'patch.edgecolor': '#CCCCCC',
    'patch.antialiased': True,

    # images
    'image.cmap': 'gist_heat',
    'image.origin': 'upper',

    # Font
    'font.size': 14.0,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.unicode_minus': False,

    # Axes
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#AAAAAA',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': 'k',
    'axes.axisbelow': True,

    # Ticks
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.color': '#565656',
    'xtick.direction': 'in',
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.color': '#565656',
    'ytick.direction': 'in',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',

    # Legend
    'legend.fancybox': True,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [8, 6],
    'figure.facecolor': '1.0',
    'figure.edgecolor': '0.50',
    'figure.subplot.hspace': 0.5,

    # Other
    'savefig.dpi': 300,
}

try:
    from cycler import cycler
    mpl_style['axes.prop_cycle'] = cycler('color', ['#1A1A1A', # black
                                                    '#2166AC', # blue
                                                    '#006837', # green
                                                    '#B2182B', # red
                                                    '#762A83', # purple
                                                    '#E08214',
                                                    '#80CDC1',
                                                    '#C51B7D',
                                                    '#FEE08B'])
except ImportError:
    mpl_style['axes.color_cycle'] = ['#1A1A1A', # black
                                     '#2166AC', # blue
                                     '#006837', # green
                                     '#B2182B', # red
                                     '#762A83', # purple
                                     '#E08214',
                                     '#80CDC1',
                                     '#C51B7D',
                                     '#FEE08B']
