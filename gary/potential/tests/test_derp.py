"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.units as u

# This project
from ..builtin import HenonHeilesPotential

def test_thing():
    print("ONE")
    potential = HenonHeilesPotential()
    print("TWO")
