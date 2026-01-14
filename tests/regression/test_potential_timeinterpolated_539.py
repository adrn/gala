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
        pot = pickle.load(f)
