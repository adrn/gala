from collections.abc import Iterable

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np

from gala.dynamics import Orbit

__all__ = ['get_staeckel_fudge_delta', 'find_actions_staeckel']


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
    a2_c2[np.abs(a2_c2) < 1e-12] = 0.  # MAGIC NUMBER / HACK
    delta = np.sqrt(a2_c2)

    # Median over time if the inputs were orbits
    if (len(delta.shape) > 1 and median) or isinstance(w, Orbit):
        delta = np.nanmedian(delta, axis=0)

    return delta * potential.units['length']


def find_actions_staeckel(potential, w, mean=True, delta=None,
                          ro=None, vo=None):
    """
    Compute approximate actions, angles, and frequencies using the Staeckel
    Fudge as implemented in Galpy. If you use this function, please also cite
    Galpy in your work (Bovy 2015).

    Parameters
    ----------
    potential : potential-like
        A Gala potential instances.
    w : `~gala.dynamics.PhaseSpacePosition` or `~gala.dynamics.Orbit`
        Either a set of initial conditions / phase-space positions, or a set of
        orbits computed in the input potential.
    mean : bool (optional)
        If an `~gala.dynamics.Orbit` is passed in, take the mean over actions
        and frequencies.
    delta : numeric, array-like (optional)
        The focal length parameter, ∆, used by the Staeckel fudge. This is
        computed if not provided.
    ro : quantity-like (optional)
    vo : quantity-like (optional)

    Returns
    -------
    aaf : `astropy.table.QTable`
        An Astropy table containing the actions, angles, and frequencies for
        each input phase-space position or orbit.

    """
    from galpy.actionAngle import actionAngleStaeckel

    if delta is None:
        delta = get_staeckel_fudge_delta(potential, w)

    galpy_potential = potential.to_galpy_potential(ro, vo)
    if isinstance(galpy_potential, list):
        ro = galpy_potential[0]._ro * u.kpc
        vo = galpy_potential[0]._vo * u.km/u.s
    else:
        ro = galpy_potential._ro * u.kpc
        vo = galpy_potential._vo * u.km/u.s

    if not isinstance(w, Orbit):
        w = Orbit(w.pos[None], w.vel[None], t=[0.] * potential.units['time'])

    if w.norbits == 1:
        iter_ = [w]
    else:
        iter_ = w.orbit_gen()

    if isinstance(delta, u.Quantity):
        delta = np.atleast_1d(delta)

    if not isinstance(delta, Iterable):
        delta = [delta] * w.norbits

    if len(delta) != w.norbits:
        raise ValueError(
            "Input delta must have same shape as the inputted number of orbits"
        )

    rows = []
    for w_, delta_ in zip(iter_, delta):
        o = w_.to_galpy_orbit(ro, vo)
        aAS = actionAngleStaeckel(pot=galpy_potential, delta=delta_)

        aaf = aAS.actionsFreqsAngles(o)
        aaf = {
            "actions": np.array(aaf[:3]).T * ro * vo,
            "freqs": np.array(aaf[3:6]).T * vo / ro,
            "angles": coord.Angle(np.array(aaf[6:]).T * u.rad),
        }
        if mean:
            aaf['actions'] = np.nanmean(aaf['actions'], axis=0)
            aaf['freqs'] = np.nanmean(aaf['freqs'], axis=0)
            aaf['angles'] = aaf['angles'][0]
        rows.append(aaf)
    return at.QTable(rows=rows)
