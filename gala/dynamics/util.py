"""General dynamics utilities."""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.utils.misc import isiterable
from scipy.signal import argrelmax, argrelmin

from ..util import atleast_2d
from .core import PhaseSpacePosition

__all__ = ["combine", "estimate_dt_n_steps", "peak_to_peak_period"]


def peak_to_peak_period(t, f, amplitude_threshold=1e-2):
    """
    Estimate the period of a time series using peak-to-peak analysis.

    This function estimates the period of an oscillating time series by
    identifying peaks and troughs and computing the mean time between them.

    Parameters
    ----------
    t : array_like
        Time grid aligned with the input time series.
    f : array_like
        A periodic time series to analyze.
    amplitude_threshold : float, optional
        A tolerance parameter for the minimum relative amplitude. The analysis
        fails if the mean amplitude of oscillations isn't larger than this
        threshold. Default is 1e-2.

    Returns
    -------
    period : float or :class:`~astropy.units.Quantity`
        The estimated period. Returns the same type as the input ``t``.
        Returns NaN if the amplitude threshold is not met.

    Notes
    -----
    This method works best for approximately sinusoidal time series with
    well-defined peaks. For irregular or noisy data, consider smoothing
    the input first.
    """
    if hasattr(t, "unit"):
        t_unit = t.unit
        t = t.value
    else:
        t_unit = u.dimensionless_unscaled

    # find peaks
    max_ix = argrelmax(f, mode="wrap")[0]
    max_ix = max_ix[(max_ix != 0) & (max_ix != (len(f) - 1))]

    # find troughs
    min_ix = argrelmin(f, mode="wrap")[0]
    min_ix = min_ix[(min_ix != 0) & (min_ix != (len(f) - 1))]

    # neglect minor oscillations
    if abs(np.mean(f[max_ix]) - np.mean(f[min_ix])) < amplitude_threshold:
        return np.nan * t_unit

    # compute mean peak-to-peak
    T_max = np.mean(t[max_ix[1:]] - t[max_ix[:-1]]) if len(max_ix) > 0 else np.nan

    # now compute mean trough-to-trough
    T_min = np.mean(t[min_ix[1:]] - t[min_ix[:-1]]) if len(min_ix) > 0 else np.nan

    # then take the mean of these two
    return np.mean([T_max, T_min]) * t_unit


def _autodetermine_initial_dt(w0, H, dE_threshold=1e-9, **integrate_kwargs):
    if w0.shape and w0.shape[0] > 1:
        raise ValueError(
            "Only one set of initial conditions may be passed in at a time."
        )

    if dE_threshold is None:
        return 1.0

    dts = np.logspace(-3, 1, 8)[::-1]
    base_n_steps = 1000

    for dt in dts:
        n_steps = round(base_n_steps / dt)
        orbit = H.integrate_orbit(w0, dt=dt, n_steps=n_steps, **integrate_kwargs)
        E = orbit.energy()
        dE = np.abs((E[-1] - E[0]) / E[0]).value

        if dE < dE_threshold:
            break

    return dt


def estimate_dt_n_steps(
    w0,
    hamiltonian,
    n_periods,
    n_steps_per_period,
    dE_threshold=1e-9,
    func=np.nanmax,
    **integrate_kwargs,
):
    """
    Estimate the timestep and number of steps for orbit integration.

    This function estimates appropriate integration parameters based on the
    orbital period and desired sampling. It first integrates a short orbit
    to determine the period, then calculates the timestep and number of
    steps needed for the requested integration time.

    Parameters
    ----------
    w0 : :class:`~gala.dynamics.PhaseSpacePosition` or array_like
        Initial conditions for the orbit.
    hamiltonian : :class:`~gala.potential.Hamiltonian` or :class:`~gala.potential.PotentialBase`
        The Hamiltonian or potential to integrate the orbit in.
    n_periods : int
        Number of (maximum) orbital periods to integrate for.
    n_steps_per_period : int
        Number of integration steps to take per (maximum) orbital period.
    dE_threshold : float, optional
        Maximum fractional energy difference used to determine initial
        timestep for the test integration. Set to ``None`` to ignore this
        constraint. Default is 1e-9.
    func : callable, optional
        Function that determines which period to use when multiple periods
        are found (e.g., for 3D orbits). Default is :func:`~numpy.nanmax`,
        which uses the maximum period. Other options include
        :func:`~numpy.nanmin`, :func:`~numpy.nanmean`, :func:`~numpy.nanmedian`.
    **integrate_kwargs
        Additional keyword arguments passed to the orbit integration method.

    Returns
    -------
    dt : float
        The recommended timestep for integration.
    n_steps : int
        The recommended number of integration steps.

    Raises
    ------
    RuntimeError
        If no period can be determined from the test orbit.
    ValueError
        If the computed timestep is zero or very small.

    Notes
    -----
    This function works by:
    1. Integrating a test orbit to determine the characteristic periods
    2. Using the specified function to select the period to use
    3. Computing dt and n_steps based on the desired sampling
    """
    if not isinstance(w0, PhaseSpacePosition):
        w0 = np.asarray(w0)
        w0 = PhaseSpacePosition.from_w(w0, units=hamiltonian.units)

    from ..potential import Hamiltonian

    hamiltonian = Hamiltonian(hamiltonian)

    # integrate orbit
    dt = _autodetermine_initial_dt(
        w0, hamiltonian, dE_threshold=dE_threshold, **integrate_kwargs
    )
    n_steps = round(10000 / dt)
    orbit = hamiltonian.integrate_orbit(w0, dt=dt, n_steps=n_steps, **integrate_kwargs)

    # if loop, align circulation with Z and take R period
    circ = orbit.circulation()
    if np.any(circ):
        orbit = orbit.align_circulation_with_z(circulation=circ)
        cyl = orbit.represent_as(coord.CylindricalRepresentation)

        # convert to cylindrical coordinates
        R = cyl.rho.value
        phi = cyl.phi.value
        z = cyl.z.value

        T = (
            np.array([peak_to_peak_period(orbit.t, f).value for f in [R, phi, z]])
            * orbit.t.unit
        )

    else:
        T = (
            np.array([peak_to_peak_period(orbit.t, f).value for f in orbit.pos])
            * orbit.t.unit
        )

    # timestep from number of steps per period
    T = func(T)

    if np.isnan(T):
        raise RuntimeError("Failed to find period.")

    T = T.decompose(hamiltonian.units).value
    dt = T / float(n_steps_per_period)
    n_steps = round(n_periods * T / dt)

    if dt == 0.0 or dt < 1e-13:
        raise ValueError("Timestep is zero or very small!")

    return dt, n_steps


def combine(objs):
    """
    Combine multiple PhaseSpacePosition or Orbit objects into a single object.

    This function concatenates multiple objects of the same type into a single
    object. All input objects must have the same type, dimensionality, reference
    frame, and (for Orbit objects) the same time array and potential.

    Parameters
    ----------
    objs : iterable
        A sequence of :class:`~gala.dynamics.PhaseSpacePosition` or
        :class:`~gala.dynamics.Orbit` objects to combine. All objects must
        be of the same type.

    Returns
    -------
    combined : :class:`~gala.dynamics.PhaseSpacePosition` or :class:`~gala.dynamics.Orbit`
        A single object containing the combined data from all input objects.
        The output will have the same type as the input objects.

    Raises
    ------
    ValueError
        If the input is empty or contains only one object, or if the objects
        have different reference frames, potentials, or time arrays.
    TypeError
        If the objects are not all of the same type, or if they are not
        PhaseSpacePosition or Orbit instances.
    NotImplementedError
        If the objects do not have Cartesian representations.

    Examples
    --------
    Combine multiple phase-space positions::

        >>> import gala.dynamics as gd
        >>> import astropy.units as u
        >>> w1 = gd.PhaseSpacePosition(pos=[1,0,0]*u.kpc, vel=[0,1,0]*u.km/u.s)
        >>> w2 = gd.PhaseSpacePosition(pos=[0,1,0]*u.kpc, vel=[1,0,0]*u.km/u.s)
        >>> combined = gd.combine([w1, w2])
        >>> combined.shape
        (2,)

    Notes
    -----
    Currently, this function only works for objects with Cartesian coordinate
    representations. The objects are combined by concatenating their position
    and velocity arrays along the appropriate axis.
    """
    from .orbit import Orbit

    # have to special-case this because they are iterable
    if isinstance(objs, PhaseSpacePosition | Orbit) or (
        not isiterable(objs) or len(objs) < 1
    ):
        raise ValueError("You must pass a non-empty iterable to combine.")

    if len(objs) == 1:  # short circuit
        return objs[0]

    # We only support these two types to combine:
    if objs[0].__class__ not in {PhaseSpacePosition, Orbit}:
        raise TypeError("Objects must be either PhaseSpacePosition or Orbit instances.")

    # Validate objects:
    # - check type
    # - check dimensionality
    # - check frame, potential
    # - Right now, we only support Cartesian
    for obj in objs:
        # Check to see if they are all the same type of object:
        if obj.__class__ != objs[0].__class__:
            raise TypeError("All objects must have the same type.")

        # Make sure they have same dimensionality
        if obj.ndim != objs[0].ndim:
            raise ValueError("All objects must have the same ndim.")

        # Check that all objects have the same reference frame
        if obj.frame != objs[0].frame:
            raise ValueError("All objects must have the same frame.")

        # Check that (for orbits) they all have the same potential
        if hasattr(obj, "potential") and obj.potential != objs[0].potential:
            raise ValueError("All objects must have the same potential.")

        # For orbits, time arrays must be the same
        if (
            hasattr(obj, "t")
            and obj.t is not None
            and objs[0].t is not None
            and not u.allclose(obj.t, objs[0].t, atol=1e-13 * objs[0].t.unit)
        ):
            raise ValueError("All orbits must have the same time array.")

        if "cartesian" not in obj.pos.get_name():
            raise NotImplementedError(
                "Currently, combine only works for Cartesian-represented objects."
            )

    # Now we prepare the positions, velocities:
    if objs[0].__class__ == PhaseSpacePosition:
        pos = []
        vel = []

        for i, obj in enumerate(objs):
            if i == 0:
                pos_unit = obj.pos.xyz.unit
                vel_unit = obj.vel.d_xyz.unit

            pos.append(atleast_2d(obj.pos.xyz.to(pos_unit).value, insert_axis=1))
            vel.append(atleast_2d(obj.vel.d_xyz.to(vel_unit).value, insert_axis=1))

        pos = np.concatenate(pos, axis=1) * pos_unit
        vel = np.concatenate(vel, axis=1) * vel_unit

        return PhaseSpacePosition(pos=pos, vel=vel, frame=objs[0].frame)

    if objs[0].__class__ == Orbit:
        pos = []
        vel = []

        for i, obj in enumerate(objs):
            if i == 0:
                pos_unit = obj.pos.xyz.unit
                vel_unit = obj.vel.d_xyz.unit

            p = obj.pos.xyz.to(pos_unit).value
            v = obj.vel.d_xyz.to(vel_unit).value

            if p.ndim < 3:
                p = p.reshape((*p.shape, 1))
                v = v.reshape((*v.shape, 1))

            pos.append(p)
            vel.append(v)

        pos = np.concatenate(pos, axis=2) * pos_unit
        vel = np.concatenate(vel, axis=2) * vel_unit

        return Orbit(
            pos=pos,
            vel=vel,
            t=objs[0].t,
            frame=objs[0].frame,
            potential=objs[0].potential,
        )

    raise RuntimeError("should never get here...")
