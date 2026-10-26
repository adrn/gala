import pickle

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import GSL_ENABLED

import gala.potential as gp


@pytest.mark.skipif(
    not GSL_ENABLED,
    reason="requires Gala compiled with GSL support",
)
def test_timeinterpolated_pickle(tmpdir):
    # construct a simple time-evolving NFW potential
    times = np.linspace(0, 10, 100) * u.Gyr
    masses = np.linspace(1e11, 5e11, 100) * u.Msun
    pot = gp.TimeInterpolatedPotential(
        gp.NFWPotential, times, m=masses, r_s=20 * u.kpc, units="galactic"
    )
    with open(tmpdir.join("time_interp_pot.pkl"), "wb") as f:
        pickle.dump(pot, f)

    with open(tmpdir.join("time_interp_pot.pkl"), "rb") as f:
        new_pot = pickle.load(f)

    q = [8.0, 1.0, 0.5] * u.kpc
    for t in [0.0, 3.3, 10.0] * u.Gyr:
        assert u.allclose(new_pot.energy(q, t=t), pot.energy(q, t=t))
        assert u.allclose(new_pot.gradient(q, t=t), pot.gradient(q, t=t))
        assert u.allclose(new_pot.density(q, t=t), pot.density(q, t=t))
