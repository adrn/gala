"""Test some builtin potentials against galpy"""

# Third-party
import numpy as np
from astropy.constants import G
import astropy.units as u
import pytest
from scipy.special import gamma

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
    HAS_GALPY = True
except ImportError:
    HAS_GALPY = False

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


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_kepler():
    from galpy.potential import KeplerPotential as BovyKeplerPotential

    M = 5e10 * u.Msun
    gala_pot = KeplerPotential(m=M, units=galactic)

    amp = (G*M).to_value(vo**2 * ro)
    bovy_pot = BovyKeplerPotential(amp=amp, ro=ro, vo=vo)

    helper(gala_pot, bovy_pot)


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_miyamoto():
    from galpy.potential import MiyamotoNagaiPotential as BovyMiyamotoNagaiPotential

    M = 5e10 * u.Msun
    gala_pot = MiyamotoNagaiPotential(m=M, a=3.5*u.kpc, b=300*u.pc,
                                      units=galactic)

    amp = (G*M).to_value(vo**2 * ro)
    a = gala_pot.parameters['a'].to_value(ro)
    b = gala_pot.parameters['b'].to_value(ro)
    bovy_pot = BovyMiyamotoNagaiPotential(amp=amp, a=a, b=b, ro=ro, vo=vo)

    helper(gala_pot, bovy_pot)


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_nfw():
    from galpy.potential import NFWPotential as BovyNFWPotential

    M = 6.5854e10 * u.Msun
    gala_pot = NFWPotential(m=M, r_s=15*u.kpc, units=galactic)
    amp = (G*M).to_value(vo**2 * ro)
    a = gala_pot.parameters['r_s'].to_value(ro)
    bovy_pot = BovyNFWPotential(amp, a=a, ro=ro, vo=vo)

    helper(gala_pot, bovy_pot)


@pytest.mark.skipif(not HAS_GALPY or not GSL_ENABLED,
                    reason="requires galpy and GSL to run this test")
def test_powerlawcutoff():
    from galpy.potential import PowerSphericalPotentialwCutoff

    M = 1.5854e10 * u.Msun
    alpha = 1.75
    gala_pot = PowerLawCutoffPotential(m=M, alpha=alpha, r_c=15*u.kpc,
                                       units=galactic)
    r_c = gala_pot.parameters['r_c']
    amp = (G*M).to_value(vo**2 * ro) * ((1/(2*np.pi) * r_c.to_value(ro)**(alpha - 3) / (gamma(3/2 - alpha/2))))
    bovy_pot = PowerSphericalPotentialwCutoff(amp,
                                              alpha=gala_pot.parameters['alpha'].value,
                                              rc=r_c.to_value(ro),
                                              ro=ro, vo=vo)

    helper(gala_pot, bovy_pot)


@pytest.mark.skipif(not HAS_GALPY or not GSL_ENABLED,
                    reason="requires galpy and GSL to run this test")
def test_mwpotential2014():
    from galpy.potential import (MWPotential2014,
                                 evaluateDensities,
                                 evaluatezforces,
                                 evaluatePotentials)

    # Here these have to be default:
    ro = 8 * u.kpc
    vo = 220 * u.km/u.s

    gala_pot = BovyMWPotential2014()
    bovy_pot = MWPotential2014
    for x in bovy_pot:
        x.turn_physical_on()

    Rs = np.random.uniform(1, 15, size=ntest) * u.kpc
    zs = np.random.uniform(1, 15, size=ntest) * u.kpc

    xyz = np.zeros((3, Rs.size)) * u.kpc
    xyz[0] = Rs
    xyz[2] = zs
    assert np.allclose(gala_pot.density(xyz).to_value(u.Msun/u.pc**3),
                       evaluateDensities(bovy_pot, R=Rs.to_value(ro), z=zs.to_value(ro)))

    assert np.allclose(gala_pot.energy(xyz).to_value((u.km / u.s)**2),
                       evaluatePotentials(bovy_pot, R=Rs.to_value(ro), z=zs.to_value(ro)))

    assert np.allclose(gala_pot.gradient(xyz).to_value((u.km/u.s) * u.pc/u.Myr / u.pc)[2],
                       -evaluatezforces(bovy_pot, R=Rs.to_value(ro), z=zs.to_value(ro)))
