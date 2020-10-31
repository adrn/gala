"""
    Test the special potentials...
"""

# Third-party
import astropy.units as u
import pytest

# This project
from .helpers import CompositePotentialTestBase
from ..builtin.special import (LM10Potential, MilkyWayPotential,
                               BovyMWPotential2014)
from ...._cconfig import GSL_ENABLED


class TestLM10Potential(CompositePotentialTestBase):
    potential = LM10Potential()
    w0 = [8., 0., 0., 0., 0.22, 0.1]


class TestLM10Potential2(CompositePotentialTestBase):
    potential = LM10Potential(disk={'m': 5E10*u.Msun}, bulge={'m': 5E10*u.Msun})
    w0 = [8., 0., 0., 0., 0.22, 0.1]


class TestMilkyWayPotential(CompositePotentialTestBase):
    potential = MilkyWayPotential()
    w0 = [8., 0., 0., 0., 0.22, 0.1]


@pytest.mark.skipif(not GSL_ENABLED,
                    reason="requires GSL to run this test")
class TestBovyMWPotential2014(CompositePotentialTestBase):
    w0 = [8., 0., 0., 0., 0.22, 0.1]

    def setup(self):
        self.potential = BovyMWPotential2014()
        super().setup()
