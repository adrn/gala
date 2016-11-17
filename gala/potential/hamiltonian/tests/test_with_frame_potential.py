# coding: utf-8

import pytest

# Project
from .. import Hamiltonian
from ...potential.builtin import SphericalNFWPotential
from ...frame.builtin import StaticFrame
from ...tests.helpers import _TestBase
from ....units import galactic

class TestLogPotentialWithStaticFrame(_TestBase):
    obj = Hamiltonian(SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic),
                      StaticFrame(units=galactic))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass
