.. include:: ../references.txt

.. _time-interpolated-potential:

*************************************
Time-dependent Potentials
*************************************

Introduction
============

Many scenarios involve gravitational potentials that change over time, for example, a
halo with a growing mass and/or scale radius, a star cluster losing mass into a stellar
stream, or galactic bar rotating with a slowing pattern speed.

The :class:`~gala.potential.potential.TimeInterpolatedPotential` class enables
modeling of such time-dependent potentials by wrapping any existing potential
class and interpolating its parameters, origin, and/or rotation matrices over
time using `GSL spline interpolation <https://www.gnu.org/software/gsl/doc/html/interp.html>`_.

.. note::

    This feature requires Gala to be compiled with GSL support.

Getting Started
===============

Use the `~gala.potential.potential.TimeInterpolatedPotential` class to wrap a potential class, and pass parameters to the wrapper either as constant values or as arrays (interpreted as the value of the parameter). For example, to specify a Hernquist potential with a time-varying mass but a constant scale radius::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> from gala.units import galactic
    >>>
    >>> # Interpolation time knots
    >>> knots = np.linspace(0, 1, 32) * u.Gyr
    >>>
    >>> # Mass grows exponentially over 1 Gyr
    >>> mass_t = np.geomspace(1e10, 2e10, 32) * u.Msun
    >>>
    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.HernquistPotential,
    ...     time_knots=knots,
    ...     m=mass_t,
    ...     c=10.0 * u.kpc,  # constant
    ...     units=galactic
    ... )

Now we can evaluate the potential at different times::

    >>> pos = [8., 0, 0] * u.kpc
    >>> pot.energy(pos, t=0 * u.Gyr)
    <Quantity [-0.00249917] kpc2 / Myr2>
    >>> pot.energy(pos, t=0.5 * u.Gyr)
    <Quantity [-0.00353436] kpc2 / Myr2>
    >>> pot.energy(pos, t=1.0 * u.Gyr)
    <Quantity [-0.00499834] kpc2 / Myr2>

The energy increases (becomes more negative) as the mass grows, as expected.

To reiterate, time-varying parameters are specified as arrays with length matching the
number of time knots, but you can mix constant and time-varying parameters freely. The
wrapper will automatically detect which parameters vary with time based on the array
shape.


Interpolation Methods
=====================

The class supports several interpolation methods from GSL:

- ``'linear'``: Linear interpolation (fastest, but not smooth). Requires at least 2 knots.
- ``'cspline'``: Cubic spline interpolation with natural boundary conditions (default). Smooth with continuous second derivatives. Requires at least 3 knots.
- ``'akima'``: Akima spline interpolation. Avoids overshoot in regions with rapidly changing curvature. Requires at least 5 knots.
- ``'steffen'``: Steffen's monotonic interpolation. Guarantees monotonicity between data points. Requires at least 3 knots.

Choose the interpolation method based on your needs. If you do not have an opinion, we
recommend using the default (cubic spline) because it has continuous second
derivatives. Specify the interpolation method via the ``interpolation_method``
argument::

    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.NFWPotential,
    ...     time_knots=knots,
    ...     m=mass_t,
    ...     r_s=20.0 * u.kpc,
    ...     interpolation_method='akima',  # Use Akima splines
    ...     units=galactic
    ... )


Time-dependent Origin and Rotation
==================================

In addition to potential parameters, you can specify a time-varying origin (center
position) or rotation matrix. This is useful for modeling moving objects or tracking a
system in a non-inertial frame.

A time-interpolated origin is specified as an array of shape ``(n_knots, 3)``::

    >>> # Origin moves in a circle
    >>> theta = np.linspace(0, 2*np.pi, len(knots))
    >>> origins = np.column_stack([
    ...     5 * np.cos(theta),
    ...     5 * np.sin(theta),
    ...     np.zeros(len(knots))
    ... ]) * u.kpc
    >>>
    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.PlummerPotential,
    ...     time_knots=knots,
    ...     m=1e10 * u.Msun,
    ...     b=1.0 * u.kpc,
    ...     origin=origins,
    ...     units=galactic
    ... )

Now the potential's center follows a circular orbit over time:

.. plot::
    :align: center
    :context: close-figs

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import gala.potential as gp
    from gala.units import galactic

    # Interpolation time knots
    knots = np.linspace(0, 1, 32) * u.Gyr

    # Mass grows exponentially over 1 Gyr
    mass_t = np.geomspace(1e10, 2e10, 32) * u.Msun

    # Origin moves in a circle
    theta = np.linspace(0, 2*np.pi, len(knots))
    origins = np.column_stack([
        5 * np.cos(theta),
        5 * np.sin(theta),
        np.zeros(len(knots))
    ]) * u.kpc
    pot = gp.TimeInterpolatedPotential(
        gp.PlummerPotential,
        time_knots=knots,
        m=1e10 * u.Msun,
        b=1.0 * u.kpc,
        origin=origins,
        units=galactic
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(6, 6), sharex=True, sharey=True, layout='constrained'
    )
    for i, t in enumerate(np.linspace(0, 1, 4) * u.Gyr):
        pot.plot_density_contours(
            grid=(np.linspace(-10, 10, 128), np.linspace(-10, 10, 128), 0.0),
            t=t,
            ax=axes.flat[i],
        )
        axes.flat[i].set_title(f't = {t.to_value(u.Gyr):.2f} Gyr')


You can also specify time-varying rotation matrices by passing in an array of shape
``(n_knots, 3, 3)``. For example, a steadily rotating bar potential::

    >>> from scipy.spatial.transform import Rotation as R
    >>>
    >>> # Rotate 90 degrees over 1 Gyr
    >>> angles = np.linspace(0, np.pi/2, 11)
    >>> rotations = np.array([
    ...     R.from_rotvec([0, 0, angle]).as_matrix()
    ...     for angle in angles
    ... ])
    >>>
    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.LongMuraliBarPotential,
    ...     time_knots=times,
    ...     m=1e11 * u.Msun,
    ...     a=5.0 * u.kpc,
    ...     b=2.0 * u.kpc,
    ...     c=1.0 * u.kpc,
    ...     R=rotations,
    ...     units=galactic
    ... )

.. plot::
    :align: center
    :context: close-figs

    from scipy.spatial.transform import Rotation as R
    # Rotate 90 degrees over 1 Gyr
    angles = np.linspace(0, np.pi/2, len(knots))
    rotations = np.array([
        R.from_rotvec([0, 0, angle]).as_matrix()
        for angle in angles
    ])
    pot = gp.TimeInterpolatedPotential(
        gp.LongMuraliBarPotential,
        time_knots=knots,
        m=1e11 * u.Msun,
        a=5.0 * u.kpc,
        b=2.0 * u.kpc,
        c=1.0 * u.kpc,
        R=rotations,
        units=galactic
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(6, 6), sharex=True, sharey=True, layout='constrained'
    )
    for i, t in enumerate(np.linspace(0, 1, 4) * u.Gyr):
        pot.plot_density_contours(
            grid=(np.linspace(-10, 10, 128), np.linspace(-10, 10, 128), 0.0),
            t=t,
            ax=axes.flat[i],
        )
        axes.flat[i].set_title(f't = {t.to_value(u.Gyr):.2f} Gyr')


Orbit Integration
=================

Time-varying potentials should work seamlessly with Gala's orbit integration
functionality. Simply pass the time-dependent potential to the integration functions::

    >>> import gala.dynamics as gd
    >>> knots = np.linspace(0, 2, 21) * u.Gyr
    >>> masses = np.linspace(1e12, 2e12, 21) * u.Msun
    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.HernquistPotential,
    ...     time_knots=knots,
    ...     m=masses,
    ...     c=10.0 * u.kpc,
    ...     units=galactic
    ... )
    >>> # Initial conditions
    >>> w0 = gp.PhaseSpacePosition(
    ...     pos=[8., 0, 0] * u.kpc,
    ...     vel=[0, 220, 0] * u.km/u.s
    ... )
    >>> orbit = gp.Hamiltonian(pot).integrate_orbit(
    ...     w0, dt=1*u.Myr, n_steps=2000
    ... )


.. plot::
    :align: center
    :context: close-figs

    import gala.dynamics as gd

    knots = np.linspace(0, 2, 21) * u.Gyr
    masses = np.linspace(1e12, 4e12, 21) * u.Msun
    pot = gp.TimeInterpolatedPotential(
        gp.HernquistPotential,
        time_knots=knots,
        m=masses,
        c=10.0 * u.kpc,
        units=galactic
    )
    w0 = gd.PhaseSpacePosition(
        pos=[8., 0, 0] * u.kpc,
        vel=[0, 220, 0] * u.km/u.s
    )
    orbit = gp.Hamiltonian(pot).integrate_orbit(
        w0, dt=1*u.Myr, n_steps=2000
    )
    fig = orbit.cylindrical.plot(["t", "rho"])
    fig.axes[0].set_ylabel("radius $R$ [kpc]")


Bounds and Extrapolation
=========================

The interpolation is only valid within the range of the specified time knots.
If you evaluate the potential outside this range, it will return ``NaN``::

    >>> times = np.linspace(0, 1, 11) * u.Gyr
    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.KeplerPotential,
    ...     time_knots=times,
    ...     m=np.linspace(1e10, 2e10, 11) * u.Msun,
    ...     units=galactic
    ... )
    >>> pot.energy([8., 0, 0] * u.kpc, t=2.0 * u.Gyr)
    <Quantity [nan] kpc2 / Myr2>

See also the API documentation for
:class:`~gala.potential.potential.TimeInterpolatedPotential`.
