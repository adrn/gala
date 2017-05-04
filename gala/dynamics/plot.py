# coding: utf-8

""" ...explain... """

from __future__ import division, print_function


# Third-party
import numpy as np

# Project
from ..util import atleast_2d

__all__ = ['plot_projections', 'plot_orbits', 'three_panel']

def _get_axes(dim, subplots_kwargs=dict()):
    """
    Parameters
    ----------
    dim : int
        Dimensionality of the orbit.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
    """

    import matplotlib.pyplot as plt

    n_panels = int(dim * (dim - 1) / 2)
    figsize = subplots_kwargs.pop('figsize', (4*n_panels, 4))
    fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                             **subplots_kwargs)

    if n_panels == 1:
        axes = [axes]

    else:
        axes = axes.flat

    return axes

def plot_projections(x, relative_to=None, autolim=True, axes=None,
                     subplots_kwargs=dict(), labels=None, plot_function=None,
                     **kwargs):
    """
    Given N-dimensional quantity, ``x``, make a figure containing 2D projections
    of all combinations of the axes.

    Parameters
    ----------
    x : array_like
        Array of values. ``axis=0`` is assumed to be the dimensionality,
        ``axis=1`` is the time axis. See :ref:`shape-conventions` for more
        information.
    relative_to : bool (optional)
        Plot the values relative to this value or values.
    autolim : bool (optional)
        Automatically set the plot limits to be something sensible.
    axes : array_like (optional)
        Array of matplotlib Axes objects.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
    labels : iterable (optional)
        List or iterable of axis labels as strings. They should correspond to
        the dimensions of the input orbit.
    plot_function : callable (optional)
        The ``matplotlib`` plot function to use. By default, this is
        :func:`~matplotlib.pyplot.scatter`, but can also be, e.g.,
        :func:`~matplotlib.pyplot.plot`.
    **kwargs
        All other keyword arguments are passed to the ``plot_function``.
        You can pass in any of the usual style kwargs like ``color=...``,
        ``marker=...``, etc.

    Returns
    -------
    fig : `~matplotlib.Figure`

    """

    # don't propagate changes back...
    x = np.array(x, copy=True)
    ndim = x.shape[0]

    # get axes object from arguments
    if axes is None:
        axes = _get_axes(dim=ndim, subplots_kwargs=subplots_kwargs)

    # if the quantities are relative
    if relative_to is not None:
        x -= relative_to

    # name of the plotting function
    plot_fn_name = plot_function.__name__

    # automatically determine limits
    if autolim:
        lims = []
        for i in range(ndim):
            max_,min_ = np.max(x[i]), np.min(x[i])
            delta = max_ - min_

            if delta == 0.:
                delta = 1.

            lims.append([min_ - delta*0.05, max_ + delta*0.05])

    k = 0
    for i in range(ndim):
        for j in range(ndim):
            if i >= j:
                continue # skip diagonal, upper triangle

            plot_func = getattr(axes[k], plot_fn_name)
            plot_func(x[i], x[j], **kwargs)

            if labels is not None:
                axes[k].set_xlabel(labels[i])
                axes[k].set_ylabel(labels[j])

            if autolim:
                axes[k].set_xlim(lims[i])
                axes[k].set_ylim(lims[j])

            k += 1

    axes[0].figure.tight_layout()
    return axes[0].figure

# --- some deprecated shite ---

def plot_orbits(x, t=None, ix=None, axes=None,
                subplots_kwargs=dict(), labels=("$x$", "$y$", "$z$"), **kwargs):
    """
    Given time series of positions, `x`, make nice plots of the orbit in
    cartesian projections.

    Parameters
    ----------
    x : array_like
        Array of positions. ``axis=0`` is assumed to be the dimensionality,
        ``axis=1`` is the time axis. See :ref:`shape-conventions` for more information.
    t : array_like (optional)
        Array of times. Only used if the input orbit is 1-dimensional.
    ix : int, array_like (optional)
        Index or array of indices of orbits to plot. For example, if `x` is an
        array of shape ``(3,1024,32)`` - 1024 timesteps for 32 orbits in 3D
        positions -- `ix` would specify which of the 32 orbits to plot.
    axes : array_like (optional)
        Array of matplotlib Axes objects.
    triangle : bool (optional)
        Make a triangle plot instead of plotting all projections in a single row.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
    labels : iterable (optional)
        List or iterable of axis labels as strings. They should correspond to the
        dimensions of the input orbit.
    **kwargs
        All other keyword arguments are passed to :func:`~matplotlib.pyplot.plot`.
        You can pass in any of the usual style kwargs like ``color=...``,
        ``marker=...``, etc.

    Returns
    -------
    fig : `~matplotlib.Figure`
    """

    x = atleast_2d(x, insert_axis=1)
    if x.ndim == 2:
        x = x[...,np.newaxis]

    # dimensionality of input orbit
    dim = x.shape[0]
    if dim > 3:
        # if orbit has more than 3 dimensions, only use the first 3
        dim = 3

    # hack in some defaults to subplots kwargs so by default share x and y axes
    if 'sharex' not in subplots_kwargs:
        subplots_kwargs['sharex'] = True

    if 'sharey' not in subplots_kwargs:
        subplots_kwargs['sharey'] = True

    axes = _get_axes(dim=dim, axes=axes, triangle=triangle,
                     subplots_kwargs=subplots_kwargs)

    if ix is not None:
        ixs = np.atleast_1d(ix)
    else:
        ixs = range(x.shape[-1])

    if dim == 3:
        for ii in ixs:
            axes[0].plot(x[0,:,ii], x[1,:,ii], **kwargs)
            axes[1].plot(x[0,:,ii], x[2,:,ii], **kwargs)
            axes[2].plot(x[1,:,ii], x[2,:,ii], **kwargs)

        if triangle:
            # HACK: until matplotlib 1.4 comes out, need this
            axes[0].set_ylim(axes[0].get_xlim())
            axes[2].set_xlim(axes[0].get_ylim())

            axes[0].set_ylabel(labels[1])
            axes[1].set_xlabel(labels[0])
            axes[1].set_ylabel(labels[2])
            axes[2].set_xlabel(labels[1])

        else:
            axes[0].set_xlabel(labels[0])
            axes[0].set_ylabel(labels[1])

            axes[1].set_xlabel(labels[0])
            axes[1].set_ylabel(labels[2])

            axes[2].set_xlabel(labels[1])
            axes[2].set_ylabel(labels[2])

        if not triangle:
            axes[0].figure.tight_layout()

    elif dim == 2:
        for ii in ixs:
            axes[0].plot(x[0,:,ii], x[1,:,ii], **kwargs)

        axes[0].set_xlabel(labels[0])
        axes[0].set_ylabel(labels[1])
        axes[0].figure.tight_layout()

    elif dim == 1:
        if t is None:
            t = np.arange(x.shape[1])

        for ii in ixs:
            axes[0].plot(t, x[0,:,ii], **kwargs)

        axes[0].set_xlabel("$t$")
        axes[0].set_ylabel(labels[0])
        axes[0].figure.tight_layout()

    return axes[0].figure

def three_panel(x, relative_to=None, autolim=True, axes=None,
                subplots_kwargs=dict(), labels=None, **kwargs):
    """
    Given 3D quantities, ``x``, make a nice three-panel or triangle plot
    of projections of the values.

    Parameters
    ----------
    x : array_like
        Array of values. ``axis=0`` is assumed to be the dimensionality,
        ``axis=1`` is the time axis. See :ref:`shape-conventions` for more information.
    relative_to : bool (optional)
        Plot the values relative to this value or values.
    autolim : bool (optional)
        Automatically set the plot limits to be something sensible.
    axes : array_like (optional)
        Array of matplotlib Axes objects.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
    labels : iterable (optional)
        List or iterable of axis labels as strings. They should correspond to the
        dimensions of the input orbit.
    **kwargs
        All other keyword arguments are passed to :func:`~matplotlib.pyplot.scatter`.
        You can pass in any of the usual style kwargs like ``color=...``,
        ``marker=...``, etc.

    Returns
    -------
    fig : `~matplotlib.Figure`
    """

    # don't propagate changes back...
    x = x.copy()

    # get axes object from arguments
    axes = _get_axes(dim=3, axes=axes, subplots_kwargs=subplots_kwargs)

    # if the quantities are relative
    if relative_to is not None:
        x -= relative_to

    axes[0].scatter(x[0], x[1], **kwargs)
    axes[1].scatter(x[0], x[2], **kwargs)
    axes[2].scatter(x[1], x[2], **kwargs)

    if labels is not None:
        axes[0].set_xlabel(labels[0])
        axes[0].set_ylabel(labels[1])

        axes[1].set_xlabel(labels[0])
        axes[1].set_ylabel(labels[2])

        axes[2].set_xlabel(labels[1])
        axes[2].set_ylabel(labels[2])

    if autolim:
        lims = []
        for i in range(3):
            mx,mi = np.max(x[i]), np.min(x[i])
            delta = mx-mi

            if delta == 0.:
                delta = 1.

            lims.append((mi-delta*0.05, mx+delta*0.05))

        axes[0].set_xlim(lims[0])
        axes[0].set_ylim(lims[1])
        axes[1].set_xlim(lims[0])
        axes[1].set_ylim(lims[2])
        axes[2].set_xlim(lims[1])
        axes[2].set_ylim(lims[2])

    axes[0].figure.tight_layout()
    return axes[0].figure
