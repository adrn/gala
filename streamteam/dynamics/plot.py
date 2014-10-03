# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['plot_orbits', 'three_panel']

def _get_axes(axes=None, triangle=False, subplots_kwargs=dict()):
    """
    Parameters
    ---------
    axes : array_like (optional)
        Array of matplotlib Axes objects.
    triangle : bool (optional)
        Make a triangle plot instead of plotting all projections in a single row.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to the matplotlib `subplots()` call.
    """

    if triangle and axes is None:
        figsize = subplots_kwargs.pop('figsize', (12,12))
        sharex = subplots_kwargs.pop('sharex', True)
        sharey = subplots_kwargs.pop('sharey', True)
        fig,axes = plt.subplots(2,2,figsize=figsize, sharex=sharex, sharey=sharey,
                                **subplots_kwargs)
        axes[0,1].set_visible(False)
        axes = axes.flat
        axes = [axes[0],axes[2],axes[3]]

    elif triangle and axes is not None:
        try:
            axes = axes.flat
        except:
            pass

        if len(axes) == 4:
            axes = [axes[0],axes[2],axes[3]]

    elif not triangle and axes is None:
        figsize = subplots_kwargs.pop('figsize', (14,5))
        fig,axes = plt.subplots(1, 3, figsize=figsize, **subplots_kwargs)

    return axes


def plot_orbits(x, ix=None, axes=None, triangle=False, subplots_kwargs=dict(), **kwargs):
    """
    Given time series of positions, `x`, make nice plots of the orbit in
    cartesian projections.

    Parameters
    ----------
    x : array_like
        Array of positions. The last axis (`axis=-1`) is assumed
        to be the dimensionality, e.g., `x.shape[-1]`. The first axis
        (`axis=0`) is assumed to be the time axis.
    ix : int, array_like (optional)
        Index or array of indices of orbits to plot. For example, if `x` is an
        array of shape (1024,32,6) -- 1024 timesteps for 32 orbits in 6D
        phase-space -- `ix` would specify which of the 32 orbits to plot.
    axes : array_like (optional)
        Array of matplotlib Axes objects.
    triangle : bool (optional)
        Make a triangle plot instead of plotting all projections in a single row.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to the matplotlib `subplots()` call.

    Other Parameters
    ----------------
    kwargs
        All other keyword arguments are passed to the matplotlib `plot()` call.
        You can pass in any of the usual style kwargs like `color=...`,
        `marker=...`, etc.
    """

    if x.ndim == 2:
        x = x[:,np.newaxis]

    axes = _get_axes(axes=axes, triangle=triangle, subplots_kwargs=subplots_kwargs)

    if ix is not None:
        ixs = np.atleast_1d(ix)
    else:
        ixs = range(x.shape[1])

    for ii in ixs:
        axes[0].plot(x[:,ii,0], x[:,ii,1], **kwargs)
        axes[1].plot(x[:,ii,0], x[:,ii,2], **kwargs)
        axes[2].plot(x[:,ii,1], x[:,ii,2], **kwargs)

    if triangle:
        # HACK: until matplotlib 1.4 comes out, need this
        axes[0].set_ylim(axes[0].get_xlim())
        axes[2].set_xlim(axes[0].get_ylim())

        axes[0].set_ylabel("Y")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        axes[2].set_xlabel("Y")

    else:
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")

        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")

    if not triangle:
        axes[0].figure.tight_layout()

    return axes[0].figure

def three_panel(q, relative_to=None, symbol=None,
                axes=None, triangle=False, subplots_kwargs=dict(), **kwargs):
    """
    Given 3D quantities, `q`, (not astropy quantities...), make nice three-panel or
    triangle plots of projections of the values.

    Parameters
    ----------
    q : array_like
        Array of values. The last axis (`axis=-1`) is assumed
        to be the dimensionality, e.g., `q.shape[-1]`.
    relative_to : bool (optional)
        Plot the values relative to this value or values.
    symbol : str (optional)
        Symbol to represent the quantity for axis labels. Can be Latex.
    axes : array_like (optional)
        Array of matplotlib Axes objects.
    triangle : bool (optional)
        Make a triangle plot instead of plotting all projections in a single row.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to the matplotlib `subplots()` call.

    Other Parameters
    ----------------
    kwargs
        All other keyword arguments are passed to the matplotlib `plot()` call.
        You can pass in any of the usual style kwargs like `color=...`,
        `marker=...`, etc.
    """

    # don't propagate changes back...
    q = q.copy()

    # change default marker
    marker = kwargs.pop('marker', '.')

    # get axes object from arguments
    axes = _get_axes(axes=axes, triangle=triangle, subplots_kwargs=subplots_kwargs)

    # if the quantities are relative
    label = None
    if relative_to is not None:
        q -= relative_to

        if symbol is not None:
            label = r"$({sym}_{{ix}} - \hat{{{sym}}}_{{ix}})/\hat{{{sym}}}_{{ix}}$".format(sym=symbol)

    else:
        if symbol is not None:
            label = r"${sym}_{{ix}}$".format(sym=symbol)

    axes[0].scatter(q[:,0], q[:,1], marker=marker, **kwargs)
    axes[1].scatter(q[:,0], q[:,2], marker=marker, **kwargs)
    axes[2].scatter(q[:,1], q[:,2], marker=marker, **kwargs)

    if label is not None:
        if triangle:
            axes[0].set_ylabel(label.format(ix=2))
            axes[1].set_xlabel(label.format(ix=1))
            axes[1].set_ylabel(label.format(ix=3))
            axes[2].set_xlabel(label.format(ix=1))

        else:
            axes[0].set_xlabel(label.format(ix=1))
            axes[0].set_ylabel(label.format(ix=2))

            axes[1].set_xlabel(label.format(ix=1))
            axes[1].set_ylabel(label.format(ix=3))

            axes[2].set_xlabel(label.format(ix=2))
            axes[2].set_ylabel(label.format(ix=3))

    if not triangle:
        axes[0].figure.tight_layout()
    return axes[0].figure
