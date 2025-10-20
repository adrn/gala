import numpy as np

__all__ = ["plot_projections"]


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
        raise ImportError("matplotlib is required for visualization.")
    import matplotlib.pyplot as plt

    if subplots_kwargs is None:
        subplots_kwargs = {}

    n_panels = int(dim * (dim - 1) / 2) if dim > 1 else 1

    subplots_kwargs.setdefault("figsize", (4 * n_panels, 4))
    subplots_kwargs.setdefault("constrained_layout", True)

    _fig, axes = plt.subplots(1, n_panels, **subplots_kwargs)

    return [axes] if n_panels == 1 else axes.flat


def plot_projections(
    x,
    relative_to=None,
    autolim=True,
    axes=None,
    subplots_kwargs=None,
    labels=None,
    plot_function=None,
    **kwargs,
):
    """
    Create 2D projections of multi-dimensional data.

    Given an N-dimensional array, this function creates a figure containing
    2D projections of all combinations of coordinate pairs. This is commonly
    used for visualizing orbits or phase-space positions.

    Parameters
    ----------
    x : array_like
        Array of values with shape ``(ndim, npoints)`` where ``ndim`` is the
        number of dimensions and ``npoints`` is the number of data points.
        See :ref:`shape-conventions` for more information.
    relative_to : array_like, optional
        Values to subtract from ``x`` before plotting. Useful for plotting
        relative to a reference position.
    autolim : bool, optional
        Automatically set sensible plot limits. Default is True.
    axes : array_like, optional
        Array of matplotlib Axes objects to plot on. If not provided,
        new axes will be created.
    subplots_kwargs : dict, optional
        Dictionary of keyword arguments passed to
        :func:`~matplotlib.pyplot.subplots` when creating new axes.
    labels : list, optional
        List of axis labels as strings corresponding to each dimension
        of the input data.
    plot_function : callable, optional
        The matplotlib plotting function to use. Default is
        :func:`~matplotlib.pyplot.plot`. Other options include
        :func:`~matplotlib.pyplot.scatter`.
    **kwargs
        Additional keyword arguments passed to the plotting function.
        Examples include ``color``, ``marker``, ``linewidth``, etc.

    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        The matplotlib figure containing the projection plots.

    Notes
    -----
    This function creates an ``(ndim*(ndim-1)/2)`` subplot grid showing
    all unique pairs of coordinate projections. For example, 3D data
    creates 3 subplots: (x,y), (x,z), and (y,z).
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

            if delta == 0.0:
                delta = 1.0

            lims.append([min_ - delta * 0.02, max_ + delta * 0.02])

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
                lims[i] = (min(lims[i][0], xlims[0]), max(lims[i][1], xlims[1]))
                lims[j] = (min(lims[j][0], ylims[0]), max(lims[j][1], ylims[1]))

                axes[k].set_xlim(lims[i])
                axes[k].set_ylim(lims[j])

            k += 1

    return axes[0].figure
