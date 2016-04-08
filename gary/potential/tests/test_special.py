# coding: utf-8
"""
    Test the special potentials...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u

# This project
from .helpers import PotentialTestBase
from ..builtin.special import *
# from ...units import galactic

class SpecialPotentialTestBase(PotentialTestBase):
    def test_repr(self):
        pass

class TestLM10Potential(SpecialPotentialTestBase):
    potential = LM10Potential()
    w0 = [8.,0.,0.,0.,0.22,0.1]

class TestLM10Potential2(PotentialTestBase):
    potential = LM10Potential(disk={'m': 5E10*u.Msun}, bulge={'m': 5E10*u.Msun})
    w0 = [8.,0.,0.,0.,0.22,0.1]

# class TestTriaxialMW(PotentialTestBase):
#     potential = TriaxialMWPotential()
#     w0 = [8.,0.,0.,0.,0.22,0.1]
