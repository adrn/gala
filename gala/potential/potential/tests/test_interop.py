"""
Test converting the builtin Potential classes to other packages
"""

# Third-party
from astropy.coordinates import CylindricalRepresentation
import astropy.units as u
import numpy as np
import pytest

# This project
import gala.potential as gp
from gala.units import galactic
from gala.tests.optional_deps import HAS_GALPY
from gala.potential.potential.interop import _gala_to_galpy

# Set these globally!
ro = 8.122 * u.kpc
vo = 245 * u.km/u.s


def pytest_generate_tests(metafunc):
    # Some magic, semi-random numbers below!
    gala_pots = []
    galpy_pots = []
    for Potential in _gala_to_galpy.keys():
        init = {}
        len_scale = 1.
        for k, par in Potential._parameters.items():
            if k == 'm':
                val = 1.43e10 * u.Msun
            elif par.physical_type == 'length':
                val = 5.12 * u.kpc * len_scale
                len_scale *= 0.5
            elif par.physical_type == 'dimensionless':
                val = 1.
            elif par.physical_type == 'speed':
                val = 201.41 * u.km/u.s
            else:
                continue

            init[k] = val

        pot = Potential(**init, units=galactic)
        galpy_pot = pot.to_galpy(ro=ro, vo=vo)

        gala_pots.append(pot)
        galpy_pots.append(galpy_pot)

    # Make a composite potential too:
    gala_pots.append(gala_pots[0] + gala_pots[1])
    galpy_pots.append([galpy_pots[0], galpy_pots[1]])

    test_names = [f'{g1.__class__.__name__}:{g2.__class__.__name__}'
                  for g1, g2 in zip(gala_pots, galpy_pots)]

    metafunc.parametrize(['gala_pot', 'galpy_pot'],
                         list(zip(gala_pots, galpy_pots)),
                         ids=test_names)


class TestGalpy:

    def setup(self):
        # Test points:
        rng = np.random.default_rng(42)
        ntest = 128

        Rs = rng.uniform(1, 15, size=ntest) * u.kpc
        phis = rng.uniform(0, 2*np.pi, size=ntest)
        zs = rng.uniform(1, 15, size=ntest) * u.kpc

        xyz = CylindricalRepresentation(Rs, phis*u.rad, zs).to_cartesian().xyz

        self.Rs = Rs
        self.phis = phis
        self.zs = zs
        self.xyz = xyz

    def test_vcirc(self, gala_pot, galpy_pot):
        tmp = self.xyz.copy()
        tmp[2] = 0.

        if (not hasattr(galpy_pot, 'vcirc')
                or isinstance(gala_pot, gp.LongMuraliBarPotential)):
            pytest.skip()

        gala_vcirc = gala_pot.circular_velocity(tmp).to_value(u.km/u.s)
        galpy_vcirc = np.array([galpy_pot.vcirc(R=RR)
                                for RR in self.Rs.to_value(ro)])
        assert np.allclose(gala_vcirc, galpy_vcirc)
