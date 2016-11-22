# coding: utf-8

import astropy.units as u
import pytest

# Project
from .. import Hamiltonian
from ...potential.builtin import SphericalNFWPotential
from ...frame.builtin import StaticFrame, ConstantRotatingFrame
from ...tests.helpers import _TestBase
from ....units import galactic
from ....dynamics import CartesianPhaseSpacePosition
from ....integrate import DOPRI853Integrator

class TestLogPotentialStaticFrame(_TestBase):
    obj = Hamiltonian(SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic),
                      StaticFrame(units=galactic))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass

class TestLogPotentialRotatingFrame(_TestBase):
    obj = Hamiltonian(SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic),
                      ConstantRotatingFrame(Omega=[0,0,0.]*u.km/u.s/u.kpc, units=galactic))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass

    def test_integrate(self):

        w0 = CartesianPhaseSpacePosition(pos=[10.,0,0.2]*u.kpc,
                                         vel=[0,0.,0.02]*u.km/u.s)
        # with pytest.raises(TypeError):
        orbit = self.obj.integrate_orbit(w0, dt=0.5, n_steps=10000)

        import matplotlib.pyplot as plt
        orbit.plot(linestyle='none', alpha=0.25)
        plt.show()
