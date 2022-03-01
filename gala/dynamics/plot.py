# Third-party
import numpy as np

__all__ = ['plot_projections']


def _get_axes(dim, subplots_kwargs=None):
    """
    Parameters
    ----------
    dim : int
        Dimensionality of the orbit.
    subplots_kwargs : dict (optional)
        Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
    """
    from gala.tests.optional_deps import HAS_MATPLOTLIB
    if not HAS_MATPLOTLIB:
        raise ImportError('matplotlib is required for visualization.')
    import matplotlib.pyplot as plt

    if subplots_kwargs is None:
        subplots_kwargs = dict()

    if dim > 1:
        n_panels = int(dim * (dim - 1) / 2)
    else:
        n_panels = 1

    subplots_kwargs.setdefault('figsize', (4*n_panels, 4))
    subplots_kwargs.setdefault('constrained_layout', True)

    fig, axes = plt.subplots(1, n_panels, **subplots_kwargs)

    if n_panels == 1:
        axes = [axes]

    else:
        axes = axes.flat

    return axes


def plot_projections(x, relative_to=None, autolim=True, axes=None,
                     subplots_kwargs=None, labels=None, plot_function=None,
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

    import matplotlib.pyplot as plt  # mpl import already checked above
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # if the quantities are relative
    if relative_to is not None:
        x -= relative_to

    # name of the plotting function
    plot_fn_name = plot_function.__name__

    # automatically determine limits
    if autolim:
        lims = []
        for i in range(ndim):
            max_, min_ = np.max(x[i]), np.min(x[i])
            delta = max_ - min_

            if delta == 0.:
                delta = 1.

            lims.append([min_ - delta*0.02, max_ + delta*0.02])

    k = 0
    for i in range(ndim):
        for j in range(ndim):
            if i >= j:
                continue  # skip diagonal, upper triangle

            plot_func = getattr(axes[k], plot_fn_name)
            plot_func(x[i], x[j], **kwargs)

            if labels is not None:
                axes[k].set_xlabel(labels[i])
                axes[k].set_ylabel(labels[j])

            if autolim:
                # ensure new limits only ever expand current axis limits
                xlims = axes[k].get_xlim()
                ylims = axes[k].get_ylim()
                lims[i] = (min(lims[i][0], xlims[0]),
                           max(lims[i][1], xlims[1]))
                lims[j] = (min(lims[j][0], ylims[0]),
                           max(lims[j][1], ylims[1]))

                axes[k].set_xlim(lims[i])
                axes[k].set_ylim(lims[j])

            k += 1

    return axes[0].figure
