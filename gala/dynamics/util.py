# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from scipy.signal import argrelmax, argrelmin

# This package
from .core import CartesianPhaseSpacePosition
from ..integrate import LeapfrogIntegrator

__all__ = ['peak_to_peak_period', 'estimate_dt_n_steps']

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

def _autodetermine_initial_dt(w0, potential, dE_threshold=1E-9, Integrator=LeapfrogIntegrator):
    if w0.shape[0] > 1:
        raise ValueError("Only one set of initial conditions may be passed in at a time.")

    if dE_threshold is None:
        return 1.

    dts = np.logspace(-3, 1, 8)[::-1]
    _base_n_steps = 1000

    for dt in dts:
        n_steps = int(round(_base_n_steps / dt))
        orbit = potential.integrate_orbit(w0, dt=dt, n_steps=n_steps, Integrator=Integrator)
        E = orbit.energy()
        dE = np.abs((E[-1] - E[0]) / E[0]).value

        if dE < dE_threshold:
            break

    return dt

def estimate_dt_n_steps(w0, potential, n_periods, n_steps_per_period, dE_threshold=1E-9,
                        func=np.nanmax):
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
        Maximum fractional energy difference -- used to determine initial timestep.
        Set to ``None`` to ignore this.
    func : callable (optional)
        Determines which period to use. By default, this takes the maximum period using
        :func:`~numpy.nanmax`. Other options could be :func:`~numpy.nanmin`,
        :func:`~numpy.nanmean`, :func:`~numpy.nanmedian`.

    Returns
    -------
    dt : float
        The timestep.
    n_steps : int
        The number of timesteps to integrate for.

    """
    if not isinstance(w0, CartesianPhaseSpacePosition):
        w0 = np.asarray(w0)
        w0 = CartesianPhaseSpacePosition.from_w(w0, units=potential.units)

    # integrate orbit
    dt = _autodetermine_initial_dt(w0, potential, dE_threshold=dE_threshold)
    n_steps = int(round(10000 / dt))
    orbit = potential.integrate_orbit(w0, dt=dt, n_steps=n_steps)

    # if loop, align circulation with Z and take R period
    circ = orbit.circulation()
    if np.any(circ):
        orbit = orbit.align_circulation_with_z(circulation=circ)
        cyl,_ = orbit.represent_as(coord.CylindricalRepresentation) # ignore velocity return

        # convert to cylindrical coordinates
        R = cyl.rho.value
        phi = cyl.phi.value
        z = cyl.z.value

        T = np.array([peak_to_peak_period(orbit.t, f).value for f in [R, phi, z]])*orbit.t.unit

    else:
        T = np.array([peak_to_peak_period(orbit.t, f).value for f in orbit.pos])*orbit.t.unit

    # timestep from number of steps per period
    T = func(T)

    if np.isnan(T):
        raise RuntimeError("Failed to find period.")

    T = T.decompose(potential.units).value
    dt = T / float(n_steps_per_period)
    n_steps = int(round(n_periods * T / dt))

    if dt == 0. or dt < 1E-13:
        raise ValueError("Timestep is zero or very small!")

    return dt, n_steps
