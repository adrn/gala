# Third party
import astropy.units as u

from ....units import UnitSystem
from ..builtin import HernquistPotential


def test_replace_units():
    usys1 = UnitSystem([u.kpc, u.Gyr, u.Msun, u.radian])
    usys2 = UnitSystem([u.pc, u.Myr, u.Msun, u.degree])

    p = HernquistPotential(m=1e10 * u.Msun, c=1.0 * u.kpc, units=usys1)
    assert p.parameters["m"].unit == usys1["mass"]
    assert p.parameters["c"].unit == usys1["length"]

    p2 = p.replace_units(usys2)
    assert p2.parameters["m"].unit == usys2["mass"]
    assert p2.parameters["c"].unit == usys2["length"]
    assert p.units == usys1
    assert p2.units == usys2
