import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G as _G
from gala._cconfig import GSL_ENABLED

import gala.potential as gp
from gala.potential.potential.io import load
from gala.potential.scf import _bfe_class
from gala.units import galactic

# HACK:
this_path = Path(__file__).parent
sys.path.insert(0, str(this_path / "../potential"))

from potential_helpers import PotentialTestBase

G = _G.decompose(galactic).value

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL", allow_module_level=True)


def test_hernquist():
    nmax = 6
    lmax = 2

    M = 1e10
    r_s = 3.5

    cos_coeff = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    sin_coeff = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    cos_coeff[0, 0, 0] = 1.0
    scf_potential = _bfe_class.SCFPotential(
        m=M, r_s=r_s, Snlm=cos_coeff, Tnlm=sin_coeff, units=galactic
    )
    # scf_potential = HackPotential(m=10., units=galactic)

    nbins = 128
    rr = np.linspace(0.1, 10.0, nbins)
    xyz = np.zeros((3, nbins))
    xyz[0] = rr * np.cos(np.pi / 4.0) * np.sin(np.pi / 4.0)
    xyz[1] = rr * np.sin(np.pi / 4.0) * np.sin(np.pi / 4.0)
    xyz[2] = rr * np.cos(np.pi / 4.0)

    hernquist = gp.HernquistPotential(m=M, c=r_s, units=galactic)

    bfe_pot = scf_potential.energy(xyz).value
    true_pot = hernquist.energy(xyz).value
    np.testing.assert_allclose(bfe_pot, true_pot)

    bfe_grad = scf_potential.gradient(xyz).value
    true_grad = hernquist.gradient(xyz).value
    np.testing.assert_allclose(bfe_grad, true_grad)


def _random_scf_potential():
    rng = np.random.default_rng(298)
    shape = (4, 4, 4)  # nmax = lmax = 3
    Snlm = rng.uniform(-0.2, 0.2, shape)
    Tnlm = rng.uniform(-0.2, 0.2, shape)
    Snlm[0, 0, 0] = 1.0
    Tnlm[:, :, 0] = 0.0
    return gp.SCFPotential(m=1.0, r_s=1.0, Snlm=Snlm, Tnlm=Tnlm, units=None)


@pytest.mark.parametrize("z", [0.7, -1.3])
def test_gradient_on_z_axis(z):
    """gala#298: the gradient on the z axis (sin(theta) = 0) is the off-axis limit."""
    pot = _random_scf_potential()
    eps = 1e-7
    grad = pot.gradient([0.0, 0.0, z]).value
    for xy in ([eps, 0], [0, eps], [-eps, -eps]):
        np.testing.assert_allclose(grad, pot.gradient([*xy, z]).value, atol=1e-6)


def test_origin():
    """gala#298: energy and gradient at r = 0 are finite. The l=0 terms are cusped at
    the origin, so the gradient there is the symmetric limit."""
    pot = _random_scf_potential()
    eps = 1e-7
    origin = np.zeros(3)

    np.testing.assert_allclose(
        pot.energy(origin).value, pot.energy([eps, 0, 0]).value, rtol=1e-5
    )

    grad = pot.gradient(origin).value
    for e in np.eye(3):
        avg = 0.5 * (pot.gradient(eps * e).value + pot.gradient(-eps * e).value)
        np.testing.assert_allclose(grad, avg, rtol=1e-5)


class TestSCFPotential(PotentialTestBase):
    nmax = 6
    lmax = 2
    Snlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    Tnlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    Snlm[0, 0, 0] = 1.0
    Snlm[2, 0, 0] = 0.5
    Snlm[4, 0, 0] = 0.25

    potential = _bfe_class.SCFPotential(
        m=1e11 * u.Msun, r_s=10 * u.kpc, Snlm=Snlm, Tnlm=Tnlm, units=galactic
    )
    w0 = [4.0, 0.7, -0.9, 0.0352238, 0.1579493, 0.02]

    skip_density = True

    def test_save_load(self, tmpdir):
        fn = str(tmpdir.join(f"{self.name}.yml"))
        self.potential.save(fn)
        p = load(fn, module=_bfe_class)
        p.energy(self.w0[: self.w0.size // 2])

    @pytest.mark.skipif(True, reason="no hessian implemented")
    def test_hessian(self):
        pass

    @pytest.mark.skip(reason="to_sympy() not implemented yet")
    def test_against_sympy(self):
        pass

    def test_compare(self):
        # skip if composite potentials
        if len(self.potential.parameters) == 0:
            return

        other = self.potential.__class__(
            units=self.potential.units, **self.potential.parameters
        )
        assert other == self.potential

        pars = self.potential.parameters.copy()
        for k in pars:
            if k != 0:
                pars[k] = pars[k] * 1.1  # fmt: skip, ruff: noqa

        other = self.potential.__class__(units=self.potential.units, **pars)
        assert other != self.potential

    def test_replace_units(self):
        H = gp.Hamiltonian(self.potential)
        H2 = gp.Hamiltonian(self.potential.replace_units(self.potential.units))

        ww = [20.0, 10, 10, 0, 0.2, 0]
        w1 = H.integrate_orbit(ww, t=np.array([0, 1.0]))[-1].w(galactic).T
        w2 = H2.integrate_orbit(ww, t=np.array([0, 1.0]))[-1].w(galactic).T

        assert np.allclose(w1, w2)
