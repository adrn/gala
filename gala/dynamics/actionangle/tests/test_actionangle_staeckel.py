# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np
import pytest

# gala
from gala.dynamics import get_staeckel_fudge_delta, PhaseSpacePosition
import gala.potential as gp
from gala.units import galactic
from gala.tests.optional_deps import HAS_GALPY


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_staeckel_fudge_delta():
    import galpy.potential as galpy_pot
    from galpy.actionAngle import estimateDeltaStaeckel

    ro = 8.1 * u.kpc
    vo = 229 * u.km/u.s

    paired_potentials = []

    # Miyamoto-Nagai
    potential = gp.MiyamotoNagaiPotential(m=6e10*u.Msun, a=3*u.kpc, b=0.3*u.kpc,
                                          units=galactic)
    amp = (G * potential.parameters['m']).to_value(vo**2 * ro)
    a = potential.parameters['a'].to_value(ro)
    b = potential.parameters['b'].to_value(ro)
    galpy_potential = galpy_pot.MiyamotoNagaiPotential(amp=amp, a=a, b=b,
                                                       ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # Hernquist
    potential = gp.HernquistPotential(m=6e10*u.Msun, c=0.3*u.kpc,
                                      units=galactic)
    amp = (G * potential.parameters['m']).to_value(vo**2 * ro)
    a = potential.parameters['c'].to_value(ro)
    galpy_potential = galpy_pot.HernquistPotential(amp=amp, a=a,
                                                   ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # NFW
    potential = gp.NFWPotential(m=6e11*u.Msun, r_s=15.6*u.kpc,
                                units=galactic)
    amp = (G * potential.parameters['m']).to_value(vo**2 * ro)
    a = potential.parameters['r_s'].to_value(ro)
    galpy_potential = galpy_pot.NFWPotential(amp=amp, a=a, ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # TEST:
    # TODO: remove the randomness here
    N = 1024
    rnd = np.random.default_rng(42)
    w = PhaseSpacePosition(pos=rnd.uniform(-10, 10, size=(3, N)) * u.kpc,
                           vel=rnd.uniform(-100, 100, size=(3, N)) * u.km/u.s)

    R = w.cylindrical.rho.to_value(ro)
    z = w.z.to_value(ro)

    for p, galpy_p in paired_potentials:
        galpy_deltas = estimateDeltaStaeckel(galpy_p, R, z,
                                             no_median=True)
        gala_deltas = get_staeckel_fudge_delta(p, w).value
        assert np.allclose(gala_deltas, galpy_deltas, atol=1e-5, rtol=1e-3)
