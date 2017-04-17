# coding: utf-8
"""
    Test the unit system.
"""

from __future__ import absolute_import, unicode_literals, division, print_function


# Third party
import astropy.units as u
from astropy.constants import G,c
import numpy as np
import pytest

# This package
from ..units import UnitSystem, DimensionlessUnitSystem

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
    usys = UnitSystem(usys)

def test_constants():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
    assert np.allclose(usys.get_constant('G'), G.decompose([u.kpc, u.Myr, u.radian, u.Msun]).value)
    assert np.allclose(usys.get_constant('c'), c.decompose([u.kpc, u.Myr, u.radian, u.Msun]).value)

def test_decompose():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km/u.s)
    q = 15.*u.km/u.s
    assert q.decompose(usys).unit == u.kpc/u.Myr # uses the core units
    assert usys.decompose(q).unit == u.km/u.s

def test_dimensionless():
    usys = DimensionlessUnitSystem()
    assert usys['dimensionless'] == u.one
    assert usys['length'] == u.one

    with pytest.raises(ValueError):
        (15*u.kpc).decompose(usys)

    with pytest.raises(ValueError):
        usys.decompose(15*u.kpc)

def test_compare():
    usys1 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas/u.yr)
    usys1_clone = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas/u.yr)

    usys2 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.kiloarcsecond/u.yr)
    usys3 = UnitSystem(u.kpc, u.Myr, u.radian, u.kg, u.mas/u.yr)

    assert usys1 == usys1_clone
    assert usys1_clone == usys1

    assert usys1 != usys2
    assert usys2 != usys1

    assert usys1 != usys3
    assert usys3 != usys1
