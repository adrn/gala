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
from gala.potential.potential.interop import galpy_to_gala_potential

# Set these globally!
ro = 8.122 * u.kpc
vo = 245 * u.km/u.s

if HAS_GALPY:
    import galpy.potential as galpy_gp

    from gala.potential.potential.interop import (
        _gala_to_galpy,
        _galpy_to_gala
    )


def pytest_generate_tests(metafunc):
    # Some magic, semi-random numbers below!
    gala_pots = []
    galpy_pots = []

    if not HAS_GALPY:
        return

    # Test the Gala -> Galpy direction
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
        galpy_pot = pot.to_galpy_potential(ro=ro, vo=vo)

        gala_pots.append(pot)
        galpy_pots.append(galpy_pot)

    # Make a composite potential too:
    gala_pots.append(gala_pots[0] + gala_pots[1])
    galpy_pots.append([galpy_pots[0], galpy_pots[1]])

    # Test the Galpy -> Gala direction
    for Potential in _galpy_to_gala.keys():
        galpy_pot = Potential(ro=ro, vo=vo)  # use defaults
        pot = galpy_to_gala_potential(galpy_pot, ro=ro, vo=vo)

        gala_pots.append(pot)
        galpy_pots.append(galpy_pot)

    test_names = [f'{g1.__class__.__name__}:{g2.__class__.__name__}'
                  for g1, g2 in zip(gala_pots, galpy_pots)]

    metafunc.parametrize(['gala_pot', 'galpy_pot'],
                         list(zip(gala_pots, galpy_pots)),
                         ids=test_names)


@pytest.mark.skipif(not HAS_GALPY,
                    reason="must have galpy installed to run these tests")
class TestGalpy:

    def setup(self):
        # Test points:
        rng = np.random.default_rng(42)
        ntest = 4

        Rs = rng.uniform(1, 15, size=ntest) * u.kpc
        phis = rng.uniform(0, 2*np.pi, size=ntest) * u.radian
        zs = rng.uniform(1, 15, size=ntest) * u.kpc

        cyl = CylindricalRepresentation(Rs, phis, zs)
        xyz = cyl.to_cartesian().xyz

        self.Rs = Rs.to_value(ro)
        self.phis = phis.to_value(u.rad)
        self.zs = zs.to_value(ro)
        self.Rpz_iter = np.array(list(zip(self.Rs, self.phis, self.zs))).copy()

        self.xyz = xyz.copy()

        Jac = np.zeros((len(cyl), 3, 3))
        Jac[:, 0, 0] = xyz[0] / cyl.rho
        Jac[:, 0, 1] = xyz[1] / cyl.rho
        Jac[:, 1, 0] = (-xyz[1] / cyl.rho**2).to_value(1 / ro)
        Jac[:, 1, 1] = (xyz[0] / cyl.rho**2).to_value(1 / ro)
        Jac[:, 2, 2] = 1.
        self.Jac = Jac

    def test_density(self, gala_pot, galpy_pot):
        if isinstance(gala_pot, gp.LogarithmicPotential):
            pytest.skip()

        gala_val = gala_pot.density(self.xyz).to_value(u.Msun / u.pc**3)
        galpy_val = np.array([galpy_gp.evaluateDensities(galpy_pot,
                                                         R=RR, z=zz, phi=pp)
                              for RR, pp, zz in self.Rpz_iter])
        assert np.allclose(gala_val, galpy_val)

    def test_energy(self, gala_pot, galpy_pot):
        gala_val = gala_pot.energy(self.xyz).to_value(u.km**2 / u.s**2)
        galpy_val = np.array([galpy_gp.evaluatePotentials(galpy_pot,
                                                          R=RR, z=zz, phi=pp)
                              for RR, pp, zz in self.Rpz_iter])

        if isinstance(gala_pot, gp.LogarithmicPotential):
            # Logarithms are weird
            gala_val -= (0.5 * gala_pot.parameters['v_c']**2 *
                         np.log(ro.value**2)).to_value((u.km / u.s)**2)

        assert np.allclose(gala_val, galpy_val)

    def test_gradient(self, gala_pot, galpy_pot):
        gala_grad = gala_pot.gradient(self.xyz)
        gala_grad = gala_grad.to_value(u.km/u.s/u.Myr)

        # TODO: Starting with galpy 1.7, this has been failing because of a
        # units issue with dPhi/dphi
        if isinstance(gala_pot, gp.LongMuraliBarPotential):
            pytest.skip()

        galpy_dR = np.array([-galpy_gp.evaluateRforces(galpy_pot,
                                                       R=RR, z=zz, phi=pp)
                            for RR, pp, zz in self.Rpz_iter])
        galpy_dp = np.array([-galpy_gp.evaluatephiforces(galpy_pot,
                                                         R=RR, z=zz, phi=pp)
                            for RR, pp, zz in self.Rpz_iter])
        galpy_dp = (galpy_dp*(u.km/u.s)**2).to_value(vo**2)

        galpy_dz = np.array([-galpy_gp.evaluatezforces(galpy_pot,
                                                       R=RR, z=zz, phi=pp)
                            for RR, pp, zz in self.Rpz_iter])
        galpy_dRpz = np.stack((galpy_dR, galpy_dp, galpy_dz),
                              axis=1)

        galpy_grad = np.einsum('nij,ni->nj', self.Jac, galpy_dRpz).T

        assert np.allclose(gala_grad, galpy_grad)

    def test_vcirc(self, gala_pot, galpy_pot):
        tmp = self.xyz.copy()
        tmp[2] = 0.

        if (not hasattr(galpy_pot, 'vcirc')
                or isinstance(gala_pot, gp.LongMuraliBarPotential)):
            pytest.skip()

        gala_vcirc = gala_pot.circular_velocity(tmp).to_value(u.km/u.s)
        galpy_vcirc = np.array([galpy_pot.vcirc(R=RR)
                                for RR, *_ in self.Rpz_iter])
        assert np.allclose(gala_vcirc, galpy_vcirc)
