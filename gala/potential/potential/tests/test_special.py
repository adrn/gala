"""
    Test the special potentials...
"""

# Third-party
import astropy.units as u

# This project
from .helpers import CompositePotentialTestBase
from ..builtin.special import (LM10Potential, MilkyWayPotential,
                               BovyMWPotential2014)

class TestLM10Potential(CompositePotentialTestBase):
    potential = LM10Potential()
    w0 = [8.,0.,0.,0.,0.22,0.1]

class TestLM10Potential2(CompositePotentialTestBase):
    potential = LM10Potential(disk={'m': 5E10*u.Msun}, bulge={'m': 5E10*u.Msun})
    w0 = [8.,0.,0.,0.,0.22,0.1]

class TestMilkyWayPotential(CompositePotentialTestBase):
    potential = MilkyWayPotential()
    w0 = [8.,0.,0.,0.,0.22,0.1]

class TestBovyMWPotential2014(CompositePotentialTestBase):
    potential = MilkyWayPotential()
    w0 = [8.,0.,0.,0.,0.22,0.1]
