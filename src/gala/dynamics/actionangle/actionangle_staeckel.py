import numpy as np

__all__ = ["get_staeckel_fudge_delta"]


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
