"""Test some builtin potentials against galpy"""

# Third-party
import numpy as np
from astropy.constants import G
import astropy.units as u
import pytest

# This project
from ...._cconfig import GSL_ENABLED
from ....units import galactic
from ..builtin import (KeplerPotential, MiyamotoNagaiPotential,
                       NFWPotential, PowerLawCutoffPotential,
                       BovyMWPotential2014)

try:
    import galpy
    import galpy.orbit
    import galpy.potential
    GALPY_INSTALLED = True
except ImportError:
    GALPY_INSTALLED = False

# Set to arbitrary values for testing
ro = 8.1 * u.kpc
vo = 240 * u.km/u.s
ntest = 128

def helper(gala_pot, galpy_pot):
    Rs = np.random.uniform(1, 15, size=ntest) * u.kpc
    zs = np.random.uniform(1, 15, size=ntest) * u.kpc

    xyz = np.zeros((3, Rs.size)) * u.kpc
    xyz[0] = Rs

    assert np.allclose(gala_pot.circular_velocity(xyz).to_value(u.km/u.s),
                       galpy_pot.vcirc(R=Rs.to_value(ro)))

    xyz[2] = zs
    assert np.allclose(gala_pot.density(xyz).to_value(u.Msun/u.pc**3),
                       galpy_pot.dens(R=Rs.to_value(ro), z=zs.to_value(ro)))

    assert np.allclose(gala_pot.energy(xyz).to_value((u.km / u.s)**2),
                       galpy_pot(R=Rs.to_value(ro), z=zs.to_value(ro)))

    assert np.allclose(gala_pot.gradient(xyz).to_value((u.km/u.s) * u.pc/u.Myr / u.pc)[2],
                       -galpy_pot.zforce(R=Rs.to_value(ro), z=zs.to_value(ro)))


@pytest.mark.skipif(not GALPY_INSTALLED,
                    reason="requires galpy to run this test")
def test_kepler():
    from galpy.potential import KeplerPotential as BovyKeplerPotential

    M = 5e10 * u.Msun
    gala_pot = KeplerPotential(m=M, units=galactic)

    amp = (G*M).to_value(vo**2 * ro)
    bovy_pot = BovyKeplerPotential(amp=amp, ro=ro, vo=vo)

    helper(gala_pot, bovy_pot)
