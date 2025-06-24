"""
Test the unit system.
"""

import itertools
import pickle

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G, c

from ..units import DimensionlessUnitSystem, SimulationUnitSystem, UnitSystem


def test_create():
    # dumb
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

    with pytest.raises(ValueError):
        UnitSystem(u.kpc, u.Myr, u.radian)  # no mass

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
    assert np.allclose(
        usys.get_constant("G"), G.decompose([u.kpc, u.Myr, u.radian, u.Msun]).value
    )
    assert np.allclose(
        usys.get_constant("c"), c.decompose([u.kpc, u.Myr, u.radian, u.Msun]).value
    )


def test_decompose():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km / u.s)
    q = 15.0 * u.km / u.s
    assert q.decompose(usys).unit == u.kpc / u.Myr  # uses the core units
    assert usys.decompose(q).unit == u.km / u.s


def test_dimensionless():
    usys = DimensionlessUnitSystem()
    assert usys["dimensionless"] == u.one
    assert usys["length"] == u.one

    with pytest.raises(ValueError):
        (15 * u.kpc).decompose(usys)

    with pytest.raises(ValueError):
        usys.decompose(15 * u.kpc)


@pytest.mark.parametrize(
    ("nu1", "nu2"),
    itertools.combinations(
        {
            "length": 15 * u.kpc,
            "mass": 1e6 * u.Msun,
            "time": 5e2 * u.Myr,
            "velocity": 150 * u.km / u.s,
        }.items(),
        2,
    ),
)
def test_simulation(nu1, nu2):
    print(nu1, nu2)
    name1, unit1 = nu1
    name2, unit2 = nu2
    usys = SimulationUnitSystem(**{name1: unit1, name2: unit2})
    assert np.isclose(usys.get_constant("G"), 1.0)

    usys = SimulationUnitSystem(**{name1: unit1, name2: unit2}, G=2.4)
    assert np.isclose(usys.get_constant("G"), 2.4)


def test_compare():
    usys1 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
    usys1_clone = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)

    usys2 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.kiloarcsecond / u.yr)
    usys3 = UnitSystem(u.kpc, u.Myr, u.radian, u.kg, u.mas / u.yr)

    assert usys1 == usys1_clone
    assert usys1_clone == usys1

    assert usys1 != usys2
    assert usys2 != usys1

    assert usys1 != usys3
    assert usys3 != usys1


def test_pickle(tmpdir):
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

    with open(tmpdir / "test.pkl", "wb") as f:
        pickle.dump(usys, f)

    with open(tmpdir / "test.pkl", "rb") as f:
        usys2 = pickle.load(f)


def test_quantity_units():
    usys = UnitSystem(5 * u.kpc, 50 * u.Myr, 1e5 * u.Msun, u.rad)

    assert np.isclose((8 * u.Myr).decompose(usys).value, 8 / 50)
    usys.get_constant("G")
