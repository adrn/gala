# coding: utf-8
"""
    Test the unit system.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third party
import astropy.units as u
from astropy.constants import G,c
import numpy as np
import pytest

# This package
from ..units import UnitSystem

def test_create():
    # dumb
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

    with pytest.raises(ValueError):
        UnitSystem(u.kpc, u.Myr, u.radian) # no mass

    with pytest.raises(ValueError):
        UnitSystem(u.kpc, u.Myr, u.Msun)

    with pytest.raises(ValueError):
        UnitSystem(u.kpc, u.radian, u.Msun)

    with pytest.raises(ValueError):
        UnitSystem(u.Myr, u.radian, u.Msun)

    usys = UnitSystem((u.kpc, u.Myr, u.radian, u.Msun))

def test_constants():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
    assert np.allclose(usys.get_constant('G'), G.decompose([u.kpc, u.Myr, u.radian, u.Msun]).value)
    assert np.allclose(usys.get_constant('c'), c.decompose([u.kpc, u.Myr, u.radian, u.Msun]).value)

def test_decompose():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km/u.s)
    q = 15.*u.km/u.s
    assert q.decompose(usys).unit == u.kpc/u.Myr # uses the core units
    assert usys.decompose(q).unit == u.km/u.s
