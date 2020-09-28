""" General dynamics utilities. """

# Third-party
import astropy.units as u
import astropy.coordinates as coord
from astropy.utils.misc import isiterable
import numpy as np
from scipy.signal import argrelmax, argrelmin

# This package
from .core import PhaseSpacePosition
from ..util import atleast_2d

__all__ = ['peak_to_peak_period', 'estimate_dt_n_steps', 'combine']


def peak_to_peak_period(t, f, amplitude_threshold=1E-2):
    """
    Estimate the period of the input time series by measuring the average
    peak-to-peak time.

    Parameters
    ----------
    t : array_like
        Time grid aligned with the input time series.
    f : array_like
        A periodic time series.
    amplitude_threshold : numeric (optional)
        A tolerance parameter. Fails if the mean amplitude of oscillations
        isn't larger than this tolerance.

    Returns
    -------
    period : float
        The mean peak-to-peak period.
    """
    if hasattr(t, 'unit'):
        t_unit = t.unit
        t = t.value
    else:
        t_unit = u.dimensionless_unscaled

    # find peaks
    max_ix = argrelmax(f, mode='wrap')[0]
    max_ix = max_ix[(max_ix != 0) & (max_ix != (len(f)-1))]

    # find troughs
    min_ix = argrelmin(f, mode='wrap')[0]
    min_ix = min_ix[(min_ix != 0) & (min_ix != (len(f)-1))]

    # neglect minor oscillations
    if abs(np.mean(f[max_ix]) - np.mean(f[min_ix])) < amplitude_threshold:
        return np.nan

    # compute mean peak-to-peak
    if len(max_ix) > 0:
        T_max = np.mean(t[max_ix[1:]] - t[max_ix[:-1]])
    else:
        T_max = np.nan

    # now compute mean trough-to-trough
    if len(min_ix) > 0:
        T_min = np.mean(t[min_ix[1:]] - t[min_ix[:-1]])
    else:
        T_min = np.nan

    # then take the mean of these two
    return np.mean([T_max, T_min]) * t_unit


def _autodetermine_initial_dt(w0, H, dE_threshold=1E-9,
                              **integrate_kwargs):
    if w0.shape and w0.shape[0] > 1:
        raise ValueError("Only one set of initial conditions may be passed "
                         "in at a time.")

    if dE_threshold is None:
        return 1.

    dts = np.logspace(-3, 1, 8)[::-1]
    _base_n_steps = 1000

    for dt in dts:
        n_steps = int(round(_base_n_steps / dt))
        orbit = H.integrate_orbit(w0, dt=dt, n_steps=n_steps,
                                  **integrate_kwargs)
        E = orbit.energy()
        dE = np.abs((E[-1] - E[0]) / E[0]).value

        if dE < dE_threshold:
            break

    return dt


def estimate_dt_n_steps(w0, hamiltonian, n_periods, n_steps_per_period,
                        dE_threshold=1E-9, func=np.nanmax,
                        **integrate_kwargs):
    """
    Estimate the timestep and number of steps to integrate an orbit for
    given its initial conditions and a potential object.

    Parameters
    ----------
    w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    potential : :class:`~gala.potential.PotentialBase`
        The potential to integrate the orbit in.
    n_periods : int
        Number of (max) orbital periods to integrate for.
    n_steps_per_period : int
        Number of steps to take per (max) orbital period.
    dE_threshold : numeric (optional)
        Maximum fractional energy difference -- used to determine initial
        timestep. Set to ``None`` to ignore this.
    func : callable (optional)
        Determines which period to use. By default, this takes the maximum
        period using :func:`~numpy.nanmax`. Other options could be
        :func:`~numpy.nanmin`, :func:`~numpy.nanmean`, :func:`~numpy.nanmedian`.

    Returns
    -------
    dt : float
        The timestep.
    n_steps : int
        The number of timesteps to integrate for.

    """
    if not isinstance(w0, PhaseSpacePosition):
        w0 = np.asarray(w0)
        w0 = PhaseSpacePosition.from_w(w0, units=hamiltonian.units)

    from ..potential import Hamiltonian
    hamiltonian = Hamiltonian(hamiltonian)

    # integrate orbit
    dt = _autodetermine_initial_dt(w0, hamiltonian, dE_threshold=dE_threshold,
                                   **integrate_kwargs)
    n_steps = int(round(10000 / dt))
    orbit = hamiltonian.integrate_orbit(w0, dt=dt, n_steps=n_steps,
                                        **integrate_kwargs)

    # if loop, align circulation with Z and take R period
    circ = orbit.circulation()
    if np.any(circ):
        orbit = orbit.align_circulation_with_z(circulation=circ)
        cyl = orbit.represent_as(coord.CylindricalRepresentation)

        # convert to cylindrical coordinates
        R = cyl.rho.value
        phi = cyl.phi.value
        z = cyl.z.value

        T = np.array([peak_to_peak_period(orbit.t, f).value
                      for f in [R, phi, z]])*orbit.t.unit

    else:
        T = np.array([peak_to_peak_period(orbit.t, f).value
                      for f in orbit.pos])*orbit.t.unit

    # timestep from number of steps per period
    T = func(T)

    if np.isnan(T):
        raise RuntimeError("Failed to find period.")

    T = T.decompose(hamiltonian.units).value
    dt = T / float(n_steps_per_period)
    n_steps = int(round(n_periods * T / dt))

    if dt == 0. or dt < 1E-13:
        raise ValueError("Timestep is zero or very small!")

    return dt, n_steps


def combine(objs):
    """Combine the specified `~gala.dynamics.PhaseSpacePosition` or
    `~gala.dynamics.Orbit` objects.

    Parameters
    ----------
    objs : iterable
        An iterable of either `~gala.dynamics.PhaseSpacePosition` or
        `~gala.dynamics.Orbit` objects.
    """
    from .orbit import Orbit

    # have to special-case this because they are iterable
    if isinstance(objs, PhaseSpacePosition) or isinstance(objs, Orbit):
        raise ValueError("You must pass a non-empty iterable to combine.")

    elif not isiterable(objs) or len(objs) < 1:
        raise ValueError("You must pass a non-empty iterable to combine.")

    elif len(objs) == 1:  # short circuit
        return objs[0]

    # We only support these two types to combine:
    if objs[0].__class__ not in [PhaseSpacePosition, Orbit]:
        raise TypeError("Objects must be either PhaseSpacePosition or Orbit "
                        "instances.")

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
        if hasattr(obj, 'potential') and obj.potential != objs[0].potential:
            raise ValueError("All objects must have the same potential.")

        # For orbits, time arrays must be the same
        if (hasattr(obj, 't') and obj.t is not None and objs[0].t is not None
                and not u.allclose(obj.t, objs[0].t,
                                   atol=1E-13*objs[0].t.unit)):
            raise ValueError("All orbits must have the same time array.")

        if 'cartesian' not in obj.pos.get_name():
            raise NotImplementedError("Currently, combine only works for "
                                      "Cartesian-represented objects.")

    # Now we prepare the positions, velocities:
    if objs[0].__class__ == PhaseSpacePosition:
        pos = []
        vel = []

        for i, obj in enumerate(objs):
            if i == 0:
                pos_unit = obj.pos.xyz.unit
                vel_unit = obj.vel.d_xyz.unit

            pos.append(atleast_2d(obj.pos.xyz.to(pos_unit).value,
                                  insert_axis=1))
            vel.append(atleast_2d(obj.vel.d_xyz.to(vel_unit).value,
                                  insert_axis=1))

        pos = np.concatenate(pos, axis=1) * pos_unit
        vel = np.concatenate(vel, axis=1) * vel_unit

        return PhaseSpacePosition(pos=pos, vel=vel, frame=objs[0].frame)

    elif objs[0].__class__ == Orbit:
        pos = []
        vel = []

        for i, obj in enumerate(objs):
            if i == 0:
                pos_unit = obj.pos.xyz.unit
                vel_unit = obj.vel.d_xyz.unit

            p = obj.pos.xyz.to(pos_unit).value
            v = obj.vel.d_xyz.to(vel_unit).value

            if p.ndim < 3:
                p = p.reshape(p.shape + (1,))
                v = v.reshape(v.shape + (1,))

            pos.append(p)
            vel.append(v)

        pos = np.concatenate(pos, axis=2) * pos_unit
        vel = np.concatenate(vel, axis=2) * vel_unit

        return Orbit(pos=pos, vel=vel,
                     t=objs[0].t, frame=objs[0].frame,
                     potential=objs[0].potential)

    else:
        raise RuntimeError("should never get here...")
