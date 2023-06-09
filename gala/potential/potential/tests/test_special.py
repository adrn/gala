"""
    Test the special potentials...
"""

# Third-party
import astropy.units as u
import pytest

from ...._cconfig import GSL_ENABLED
from ..builtin.special import (
    BovyMWPotential2014,
    LM10Potential,
    MilkyWayPotential,
    MilkyWayPotential2022,
)

# This project
from .helpers import CompositePotentialTestBase


class TestLM10Potential(CompositePotentialTestBase):
    potential = LM10Potential()
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]


class TestLM10Potential2(CompositePotentialTestBase):
    potential = LM10Potential(disk={"m": 5e10 * u.Msun}, bulge={"m": 5e10 * u.Msun})
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]


class TestMilkyWayPotential(CompositePotentialTestBase):
    potential = MilkyWayPotential()
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]


class TestMilkyWayPotential2022(CompositePotentialTestBase):
    potential = MilkyWayPotential2022()
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
class TestBovyMWPotential2014(CompositePotentialTestBase):
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    check_finite_at_origin = False

    def setup_method(self):
        self.potential = BovyMWPotential2014()
        super().setup_method()

    check_finite_at_origin = False
