"""
Test the special potentials...
"""

import astropy.units as u
import pytest
from gala._cconfig import GSL_ENABLED
from potential_helpers import CompositePotentialTestBase

from gala.potential import (
    BovyMWPotential2014,
    LM10Potential,
    MilkyWayPotential,
)


class TestLM10Potential(CompositePotentialTestBase):
    potential = LM10Potential()
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    check_zero_at_infinity = False

    num_dx = 1e-3
    skip_density = True


class TestLM10Potential2(CompositePotentialTestBase):
    potential = LM10Potential(disk={"m": 5e10 * u.Msun}, bulge={"m": 5e10 * u.Msun})
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    check_zero_at_infinity = False

    num_dx = 1e-3
    skip_density = True


class TestMilkyWayPotentialv1(CompositePotentialTestBase):
    potential = MilkyWayPotential(version="v1")
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]


class TestMilkyWayPotentialv2(CompositePotentialTestBase):
    potential = MilkyWayPotential(version="v2")
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
class TestBovyMWPotential2014(CompositePotentialTestBase):
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    check_finite_at_origin = False

    def setup_method(self):
        self.potential = BovyMWPotential2014()
        super().setup_method()
