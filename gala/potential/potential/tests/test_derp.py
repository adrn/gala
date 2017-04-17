# coding: utf-8
"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function


# Third-party
import numpy as np
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import pytest

# This project
from ..core import CompositePotential
from ..builtin import HenonHeilesPotential

def test_derp():
    p = HenonHeilesPotential()
    print(p.energy([0.77, 0.31]))
    print(p.density([0.77, 0.31]))
