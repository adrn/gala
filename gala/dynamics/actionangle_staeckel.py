import numpy as np

__all__ = ['get_staeckel_fudge_delta']


def get_staeckel_fudge_delta(potential, w, median=True):
    """Estimate the focal length parameter, ∆, used by the Staeckel fudge.

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase` subclass
        The potential that the orbits were computed in, or that you would like
        to estimate the best-fitting Staeckel potential for.
    w : `~gala.dynamics.Orbit`, `~gala.dynamics.PhaseSpacePosition`
        The orbit(s) or phase space position(s) to estimate the focal length

    Returns
    -------
    deltas : `~astropy.units.Quantity` [length]
        The focal length values.

    """
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

    d2Phi_dR2 = (cosphi**2 * hess[0, 0] +
                 sinphi**2 * hess[1, 1] +
                 sin2phi * hess[0, 1])
    d2Phi_dz2 = hess[2, 2]
    d2Phi_dRdz = cosphi * hess[0, 2] + sinphi * hess[1, 2]

    # numerator of term in eq. 9 (Sanders 2012), but from Galpy,
    #   which claims there is a sign error in the manuscript??
    num = 3*z * dPhi_dR - 3*R * dPhi_dz + R*z * (d2Phi_dR2 - d2Phi_dz2)
    a2_c2 = z**2 - R**2 + num / d2Phi_dRdz
    a2_c2[np.abs(a2_c2) < 1e-12] = 0.  # MAGIC NUMBER
    delta = np.sqrt(a2_c2)

    # Median over time if the inputs were orbits
    if len(delta.shape) > 1 and median:
        delta = np.nanmedian(delta, axis=1)

    return delta * potential.units['length']
