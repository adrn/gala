import astropy.table as at
import astropy.units as u
import numpy as np

__all__ = ["find_actions_staeckel", "get_staeckel_fudge_delta"]


def find_actions_staeckel(potential, w, mean=True, delta=None):
    """
    Compute approximate actions, angles, and frequencies using the Staeckel fudge.

    If you use this functionality, please cite: Binney 2012, Sanders 2012, Vasiliev 2019

    The potential must be axisymmetric about the z axis and implemented in C.

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        An axisymmetric, C-implemented potential.
    w : :class:`~gala.dynamics.PhaseSpacePosition` or :class:`~gala.dynamics.Orbit`
        Either a set of phase-space positions, or a set of orbits computed in the
        input potential.
    mean : bool (optional)
        If an :class:`~gala.dynamics.Orbit` is passed in, take the mean over the
        actions and frequencies, and return the angles at the first time step.
    delta : quantity-like, array-like (optional)
        The focal length parameter, Δ, of the prolate spheroidal coordinate system.
        It may be a scalar, or have one value per orbit or per phase-space
        position. If not provided, this is estimated with
        :func:`get_staeckel_fudge_delta` (the median along each orbit, if ``w``
        is an Orbit). Non-positive or non-finite values use a near-spherical
        coordinate system.

    Returns
    -------
    aaf : :class:`~astropy.table.QTable`
        A table with columns ``actions`` (J_R, J_φ, J_z), ``angles`` (θ_R, θ_φ,
        θ_z), and ``freqs`` (Ω_R, Ω_φ, Ω_z), with one row per phase-space position
        or orbit. Unbound points have NaN for all values except J_φ.

    References
    ----------
    * Binney, J. 2012, MNRAS, 426, 1324
    * Sanders, J. L. 2012, MNRAS, 426, 128
    * Vasiliev, E. 2019, MNRAS, 482, 1525
    """
    from gala._cconfig import GSL_ENABLED
    from gala.dynamics import Orbit

    if not GSL_ENABLED:
        msg = "find_actions_staeckel requires gala to be built with GSL support."
        raise RuntimeError(msg)

    if not hasattr(potential, "c_instance"):
        msg = "find_actions_staeckel requires a potential implemented in C."
        raise TypeError(msg)

    from ._staeckel import staeckel_fudge_c

    usys = potential.units
    if delta is None:
        delta = get_staeckel_fudge_delta(potential, w)
    delta = u.Quantity(delta, usys["length"]).value

    xyz = w.xyz.decompose(usys).value
    vxyz = w.v_xyz.decompose(usys).value
    is_orbit = isinstance(w, Orbit)
    if is_orbit and xyz.ndim == 2:  # single orbit: add an orbit axis
        xyz = xyz[..., None]
        vxyz = vxyz[..., None]
    elif xyz.ndim == 1:  # single phase-space position
        xyz = xyz[:, None]
        vxyz = vxyz[:, None]
    shape = xyz.shape[1:]
    delta = np.broadcast_to(delta, shape)

    # The C code evaluates the potential at (R, 0, z), so check axisymmetry there
    # TODO: check symmetry instead
    R = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
    Phi = potential.energy(xyz * usys["length"]).decompose(usys).value
    Phi_axi = (
        potential.energy(np.stack((R, np.zeros_like(R), xyz[2])) * usys["length"])
        .decompose(usys)
        .value
    )
    if not np.allclose(Phi, Phi_axi, rtol=1e-8, atol=0):
        msg = "find_actions_staeckel requires a potential that is axisymmetric about z."
        raise ValueError(msg)

    ww = np.ascontiguousarray(np.concatenate((xyz, vxyz)).reshape(6, -1).T)
    out = staeckel_fudge_c(
        potential, ww, np.ascontiguousarray(delta.reshape(-1), dtype=float)
    ).reshape(*shape, 9)

    # ORder is R, phi, z for each:
    actions = out[..., 0:3]
    angles = out[..., 3:6]
    freqs = out[..., 6:9]
    if is_orbit:  # (ntimes, norbits, 3)
        if mean:
            actions = np.nanmean(actions, axis=0)
            freqs = np.nanmean(freqs, axis=0)
            angles = angles[0]
        else:
            actions, angles, freqs = (
                np.swapaxes(x, 0, 1) for x in (actions, angles, freqs)
            )

    return at.QTable(
        {
            "actions": actions * usys["length"] ** 2 / usys["time"],
            "angles": angles * u.rad,
            "freqs": freqs * u.rad / usys["time"],
        }
    )


def get_staeckel_fudge_delta(potential, w, median=True):
    """
    Estimate the focal length parameter for the Staeckel approximation.

    This function computes the focal length parameter Δ (delta) used in the
    Staeckel fudge approximation method for computing actions in axisymmetric
    potentials. The parameter is estimated using equation (9) from Sanders (2012).

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        The gravitational potential in which the orbits were computed, or
        for which you want to estimate the best-fitting Staeckel potential.
    w : :class:`~gala.dynamics.Orbit` or :class:`~gala.dynamics.PhaseSpacePosition`
        The orbit(s) or phase-space position(s) to use for estimating the
        focal length parameter.
    median : bool, optional
        If True and ``w`` is an Orbit, return the median value over the
        orbit. If False, return the full time series. Default is True.

    Returns
    -------
    delta : :class:`~astropy.units.Quantity`
        The focal length parameter(s) with units of length. If ``median=True``
        and the input is an orbit, returns a scalar or array with shape
        matching the number of orbits. If ``median=False``, returns an array
        with the same time dimension as the input orbit(s).

    Notes
    -----
    The Staeckel fudge approximation assumes that the gravitational potential
    can be approximated by a Staeckel potential in prolate spheroidal
    coordinates. This focal length parameter determines the shape of the
    coordinate system used in the approximation.

    References
    ----------
    * Sanders, J. L. 2012, MNRAS, 426, 128
    """
    from gala.dynamics import Orbit

    grad = potential.gradient(w).decompose(potential.units).value
    hess = potential.hessian(w).decompose(potential.units).value

    # avoid constructing the full jacobian:
    cyl = w.cylindrical
    R = cyl.rho.decompose(potential.units).value
    z = w.z.decompose(potential.units).value
    cosphi = np.cos(cyl.phi)
    sinphi = np.sin(cyl.phi)
    sin2phi = np.sin(2 * cyl.phi)

    # These expressions transform the Hessian in Cartesian coordinates to the
    # pieces we need in cylindrical coordinates
    # - See: gala-notebooks/Delta-Staeckel.ipnyb
    dPhi_dR = cosphi * grad[0] + sinphi * grad[1]
    dPhi_dz = grad[2]

    d2Phi_dR2 = cosphi**2 * hess[0, 0] + sinphi**2 * hess[1, 1] + sin2phi * hess[0, 1]
    d2Phi_dz2 = hess[2, 2]
    d2Phi_dRdz = cosphi * hess[0, 2] + sinphi * hess[1, 2]

    # numerator of term in eq. 9 (Sanders 2012), but from Galpy,
    #   which claims there is a sign error in the manuscript??
    num = 3 * z * dPhi_dR - 3 * R * dPhi_dz + R * z * (d2Phi_dR2 - d2Phi_dz2)
    a2_c2 = z**2 - R**2 + num / d2Phi_dRdz
    a2_c2[np.abs(a2_c2) < 1e-12] = 0.0  # MAGIC NUMBER / HACK
    delta = np.sqrt(a2_c2)

    # Median over time if the inputs were orbits
    if (len(delta.shape) > 1 and median) or isinstance(w, Orbit):
        delta = np.nanmedian(delta, axis=0)

    return delta * potential.units["length"]
