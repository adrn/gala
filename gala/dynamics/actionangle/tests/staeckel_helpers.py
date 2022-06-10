from collections.abc import Iterable

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np

from gala.dynamics import Orbit

from ..actionangle_staeckel import get_staeckel_fudge_delta


__all__ = ["galpy_find_actions_staeckel"]


def galpy_find_actions_staeckel(
    potential, w, mean=True, delta=None, ro=None, vo=None
):
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
        vo = galpy_potential[0]._vo * u.km / u.s
    else:
        ro = galpy_potential._ro * u.kpc
        vo = galpy_potential._vo * u.km / u.s

    if not isinstance(w, Orbit):
        w = Orbit(w.pos[None], w.vel[None], t=[0.0] * potential.units["time"])

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
            aaf["actions"] = np.nanmean(aaf["actions"], axis=0)
            aaf["freqs"] = np.nanmean(aaf["freqs"], axis=0)
            aaf["angles"] = aaf["angles"][0]
        rows.append(aaf)
    return at.QTable(rows=rows)
