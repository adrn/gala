import astropy.units as u
import numpy as np
from scipy.signal import argrelmin

from .core import PhaseSpacePosition
from .orbit import Orbit

__all__ = ["fast_lyapunov_max", "lyapunov_max", "surface_of_section"]


def fast_lyapunov_max(
    w0,
    hamiltonian,
    dt,
    n_steps,
    d0=1e-5,
    n_steps_per_pullback=10,
    noffset_orbits=2,
    t1=0.0,
    atol=1e-10,
    rtol=1e-10,
    nmax=0,
    return_orbit=True,
):
    """
    Compute the maximum Lyapunov exponent using a fast C-implemented method.

    This function estimates the maximum Lyapunov exponent by integrating
    the orbit along with several nearby offset orbits and tracking their
    separation over time. It uses the DOPRI853 integrator for high accuracy.

    Parameters
    ----------
    w0 : :class:`~gala.dynamics.PhaseSpacePosition` or array_like
        Initial conditions for the primary orbit.
    hamiltonian : :class:`~gala.potential.Hamiltonian`
        The Hamiltonian system to integrate in. Must contain a C-implemented
        potential and frame.
    dt : float
        Integration timestep.
    n_steps : int
        Number of integration steps to run.
    d0 : float, optional
        Initial separation between the primary orbit and offset orbits.
        Default is 1e-5.
    n_steps_per_pullback : int, optional
        Number of integration steps between each renormalization of the
        offset vectors. Default is 10.
    noffset_orbits : int, optional
        Number of offset orbits to use. Default is 2.
    t1 : float, optional
        Initial time. Default is 0.0.
    atol : float, optional
        Absolute tolerance for the integrator. Default is 1e-10.
    rtol : float, optional
        Relative tolerance for the integrator. Default is 1e-10.
    nmax : int, optional
        Maximum number of function evaluations. Default is 0 (no limit).
    return_orbit : bool, optional
        Whether to return the integrated orbit along with the Lyapunov
        exponent. Default is True.

    Returns
    -------
    LEs : :class:`~astropy.units.Quantity`
        The Lyapunov exponents calculated from each offset orbit.
    orbit : :class:`~gala.dynamics.Orbit`, optional
        The integrated primary orbit (returned only if ``return_orbit=True``).

    Raises
    ------
    TypeError
        If the Hamiltonian does not contain C-implemented components.
    ValueError
        If trying to compute Lyapunov exponents for multiple orbits
        simultaneously.

    Notes
    -----
    The Lyapunov exponent quantifies the rate of exponential divergence
    of nearby trajectories in phase space. Positive values indicate
    chaotic motion, while zero or negative values suggest regular motion.

    This implementation is optimized for speed and uses C code with the
    DOPRI853 adaptive Runge-Kutta integrator.
    """
    from gala.potential import PotentialBase

    from .lyapunov import dop853_lyapunov_max, dop853_lyapunov_max_dont_save

    # TODO: remove in v1.0
    if isinstance(hamiltonian, PotentialBase):
        from ..potential import Hamiltonian

        hamiltonian = Hamiltonian(hamiltonian)

    if not hamiltonian.c_enabled:
        raise TypeError(
            "Input Hamiltonian must contain a C-implemented potential and frame."
        )

    if not isinstance(w0, PhaseSpacePosition):
        w0 = np.asarray(w0)
        ndim = w0.shape[0] // 2
        w0 = PhaseSpacePosition(pos=w0[:ndim], vel=w0[ndim:])

    w0_ = np.squeeze(w0.w(hamiltonian.units))
    if w0_.ndim > 1:
        raise ValueError("Can only compute fast Lyapunov exponent for a single orbit.")

    if return_orbit:
        t, w, l = dop853_lyapunov_max(
            hamiltonian,
            w0_,
            dt,
            n_steps + 1,
            t1,
            d0,
            n_steps_per_pullback,
            noffset_orbits,
            atol,
            rtol,
            nmax,
        )
        w = np.rollaxis(w, -1)

        try:
            tunit = hamiltonian.units["time"]
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled

        orbit = Orbit.from_w(
            w=w, units=hamiltonian.units, t=t * tunit, hamiltonian=hamiltonian
        )
        return l / tunit, orbit

    l = dop853_lyapunov_max_dont_save(
        hamiltonian,
        w0_,
        dt,
        n_steps + 1,
        t1,
        d0,
        n_steps_per_pullback,
        noffset_orbits,
        atol,
        rtol,
        nmax,
    )

    try:
        tunit = hamiltonian.units["time"]
    except (TypeError, AttributeError):
        tunit = u.dimensionless_unscaled

    return l / tunit


def lyapunov_max(
    w0,
    integrator,
    dt,
    n_steps,
    d0=1e-5,
    n_steps_per_pullback=10,
    noffset_orbits=8,
    t1=0.0,
    units=None,
    rng=None,
):
    """

    Compute the maximum Lyapunov exponent of an orbit by integrating many
    nearby orbits (``noffset``) separated with isotropically distributed
    directions but the same initial deviation length, ``d0``. This algorithm
    re-normalizes the offset orbits every ``n_steps_per_pullback`` steps.

    Parameters
    ----------
    w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    integrator : `~gala.integrate.Integrator`
        An instantiated `~gala.integrate.Integrator` object. Must have a run() method.
    dt : numeric
        Timestep.
    n_steps : int
        Number of steps to run for.
    d0 : numeric (optional)
        The initial separation.
    n_steps_per_pullback : int (optional)
        Number of steps to run before re-normalizing the offset vectors.
    noffset_orbits : int (optional)
        Number of offset orbits to run.
    t1 : numeric (optional)
        Time of initial conditions. Assumed to be t=0.
    units : `~gala.units.UnitSystem` (optional)
        If passing in an array (not a `~gala.dynamics.PhaseSpacePosition`),
        you must specify a unit system.
    rng : `numpy.random.Generator` (optional)
        If provided, will use this random number generator to generate the
        initial offset vectors. If not provided, will use the default
        ``numpy.random.default_rng()``.

    Returns
    -------
    LEs : :class:`~astropy.units.Quantity`
        Lyapunov exponents calculated from each offset / deviation orbit.
    orbit : `~gala.dynamics.Orbit`
    """

    if units is not None:
        pos_unit = units["length"]
        vel_unit = units["length"] / units["time"]
    else:
        pos_unit = u.dimensionless_unscaled
        vel_unit = u.dimensionless_unscaled

    if not isinstance(w0, PhaseSpacePosition):
        w0 = np.asarray(w0)
        ndim = w0.shape[0] // 2
        w0 = PhaseSpacePosition(pos=w0[:ndim] * pos_unit, vel=w0[ndim:] * vel_unit)

    w0_ = w0.w(units)
    ndim = 2 * w0.ndim

    # number of iterations
    niter = n_steps // n_steps_per_pullback

    # define offset vectors to start the offset orbits on
    rng = rng or np.random.default_rng()
    d0_vec = rng.uniform(size=(ndim, noffset_orbits))
    d0_vec /= np.linalg.norm(d0_vec, axis=0)[np.newaxis]
    d0_vec *= d0

    w_offset = w0_ + d0_vec
    all_w0 = np.hstack((w0_, w_offset))

    # array to store the full, main orbit
    full_w = np.zeros((ndim, n_steps + 1, noffset_orbits + 1))
    full_w[:, 0] = all_w0
    full_ts = np.zeros((n_steps + 1,))
    full_ts[0] = t1

    # arrays to store the Lyapunov exponents and times
    LEs = np.zeros((niter, noffset_orbits))
    ts = np.zeros_like(LEs)
    time = t1
    total_steps_taken = 0
    for i in range(1, niter + 1):
        ii = i * n_steps_per_pullback

        orbit = integrator(all_w0, dt=dt, n_steps=n_steps_per_pullback, t1=time)
        tt = orbit.t.value
        ww = orbit.w(units)
        time += dt * n_steps_per_pullback

        main_w = ww[:, -1, 0:1]
        d1 = ww[:, -1, 1:] - main_w
        d1_mag = np.linalg.norm(d1, axis=0)

        LEs[i - 1] = np.log(d1_mag / d0)
        ts[i - 1] = time

        w_offset = ww[:, -1, 0:1] + d0 * d1 / d1_mag[np.newaxis]
        all_w0 = np.hstack((ww[:, -1, 0:1], w_offset))

        full_w[:, (i - 1) * n_steps_per_pullback + 1 : ii + 1] = ww[:, 1:]
        full_ts[(i - 1) * n_steps_per_pullback + 1 : ii + 1] = tt[1:]

        total_steps_taken += n_steps_per_pullback

    LEs = np.array([LEs[:ii].sum(axis=0) / ts[ii - 1] for ii in range(1, niter)])

    try:
        t_unit = units["time"]
    except (TypeError, AttributeError):
        t_unit = u.dimensionless_unscaled

    orbit = Orbit.from_w(
        w=full_w[:, :total_steps_taken],
        units=units,
        t=full_ts[:total_steps_taken] * t_unit,
    )
    return LEs / t_unit, orbit


def surface_of_section(orbit, constant_idx, constant_val=0.0):
    """
    Generate and return a surface of section from the given orbit.

    Parameters
    ----------
    orbit : `~gala.dynamics.Orbit`
        The input orbit to generate a surface of section for.
    constant_idx : int
        Integer that represents the coordinate to record crossings in. For
        example, for a 2D Hamiltonian where you want to make a SoS in
        :math:`y-p_y`, you would specify ``constant_idx=0`` (crossing the
        :math:`x` axis), and this will only record crossings for which
        :math:`p_x>0`.

    Returns
    -------
    sos : numpy ndarray

    TODO:
    - Implement interpolation to get the other phase-space coordinates truly
      at the plane, instead of just at the orbital position closest to the
      plane.

    """

    if orbit.norbits > 1:
        raise NotImplementedError("Not yet implemented, sorry!")

    w = [getattr(orbit, x) for x in orbit.pos_components] + [
        getattr(orbit, v) for v in orbit.vel_components
    ]

    ndim = orbit.ndim

    p_ix = constant_idx + ndim

    # record position on specified plane when orbit crosses
    cross_idx = argrelmin((w[constant_idx] - constant_val) ** 2)[0]
    cross_idx = cross_idx[w[p_ix][cross_idx] > 0.0]

    sos_pos = [w[i][cross_idx] for i in range(ndim)]
    sos_pos = orbit.pos.__class__(*sos_pos)

    sos_vel = [w[i][cross_idx] for i in range(ndim, 2 * ndim)]
    sos_vel = orbit.vel.__class__(*sos_vel)

    return Orbit(sos_pos, sos_vel)
