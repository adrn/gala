# coding: utf-8

# Third-party
import astropy.units as u
from astropy.constants import G as _G
import numpy as np
import pytest

# Project
from gala._cconfig import GSL_ENABLED
from gala.units import galactic
import gala.potential as gp
from gala.potential.potential.tests.helpers import PotentialTestBase
from gala.potential.potential.io import load
from .. import _bfe_class

G = _G.decompose(galactic).value

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL",
                allow_module_level=True)


def test_hernquist():
    nmax = 6
    lmax = 2

    M = 1E10
    r_s = 3.5

    cos_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    sin_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    cos_coeff[0, 0, 0] = 1.
    scf_potential = _bfe_class.SCFPotential(m=M, r_s=r_s,
                                            Snlm=cos_coeff, Tnlm=sin_coeff,
                                            units=galactic)
    # scf_potential = HackPotential(m=10., units=galactic)

    nbins = 128
    rr = np.linspace(0.1, 10., nbins)
    xyz = np.zeros((3, nbins))
    xyz[0] = rr * np.cos(np.pi/4.) * np.sin(np.pi/4.)
    xyz[1] = rr * np.sin(np.pi/4.) * np.sin(np.pi/4.)
    xyz[2] = rr * np.cos(np.pi/4.)

    hernquist = gp.HernquistPotential(m=M, c=r_s, units=galactic)

    bfe_pot = scf_potential.energy(xyz).value
    true_pot = hernquist.energy(xyz).value
    np.testing.assert_allclose(bfe_pot, true_pot)

    bfe_grad = scf_potential.gradient(xyz).value
    true_grad = hernquist.gradient(xyz).value
    np.testing.assert_allclose(bfe_grad, true_grad)


class TestSCFPotential(PotentialTestBase):
    nmax = 6
    lmax = 2
    Snlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Tnlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Snlm[0, 0, 0] = 1.
    Snlm[2, 0, 0] = 0.5
    Snlm[4, 0, 0] = 0.25

    potential = _bfe_class.SCFPotential(m=1E11*u.Msun, r_s=10*u.kpc,
                                        Snlm=Snlm, Tnlm=Tnlm, units=galactic)
    w0 = [4.0, 0.7, -0.9, 0.0352238, 0.1579493, 0.02]

    def test_save_load(self, tmpdir):
        fn = str(tmpdir.join("{}.yml".format(self.name)))
        self.potential.save(fn)
        p = load(fn, module=_bfe_class)
        p.energy(self.w0[:self.w0.size//2])

    @pytest.mark.skipif(True, reason='no hessian implemented')
    def test_hessian(self):
        pass

    @pytest.mark.skip(reason="to_sympy() not implemented yet")
    def test_against_sympy(self):
        pass

    def test_compare(self):
        # skip if composite potentials
        if len(self.potential.parameters) == 0:
            return

        other = self.potential.__class__(units=self.potential.units, **self.potential.parameters)
        assert other == self.potential

        pars = self.potential.parameters.copy()
        for k in pars.keys():
            if k != 0:
                pars[k] = 1.1 * pars[k]

        other = self.potential.__class__(units=self.potential.units, **pars)
        assert other != self.potential

    def test_replace_units(self):
        H = gp.Hamiltonian(self.potential)
        H2 = gp.Hamiltonian(self.potential.replace_units(self.potential.units))

        ww = [20., 10, 10, 0, 0.2, 0]
        w1 = H.integrate_orbit(ww, t=np.array([0, 1.]))[-1].w(galactic).T
        w2 = H2.integrate_orbit(ww, t=np.array([0, 1.]))[-1].w(galactic).T

        assert np.allclose(w1, w2)
