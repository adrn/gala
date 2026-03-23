"""
Regression test: PhaseSpacePosition and Orbit should reject velocity arguments
that have non-velocity units (e.g., length units like kpc).

See: https://github.com/adrn/gala/issues/XXX
"""

import astropy.units as u
import numpy as np
import pytest

from gala.dynamics import Orbit, PhaseSpacePosition

unit_cases = [
    # (pos_unit, vel_unit, should_raise)
    (u.kpc, u.km / u.s, False),
    (u.kpc, u.kpc / u.Myr, False),
    (u.pc, u.km / u.s, False),
    (u.one, u.one, False),
    (u.deg, u.km / u.s, True),
    (u.one, u.km / u.s, True),
    (u.kpc, u.one, True),
    (u.kpc, u.kpc, True),
    (u.kpc, u.Myr, True),
    (u.kpc, u.deg, True),
]


@pytest.mark.parametrize(("pos_unit", "vel_unit", "should_raise"), unit_cases)
def test_psp_velocity_units(pos_unit, vel_unit, should_raise):
    pos = np.random.random(size=(3, 10)) * pos_unit
    vel = np.random.random(size=(3, 10)) * vel_unit

    if should_raise:
        with pytest.raises(u.UnitTypeError):
            PhaseSpacePosition(pos=pos, vel=vel)
    else:
        PhaseSpacePosition(pos=pos, vel=vel)


@pytest.mark.parametrize(("pos_unit", "vel_unit", "should_raise"), unit_cases)
def test_orbit_velocity_units(pos_unit, vel_unit, should_raise):
    pos = np.random.random(size=(3, 10)) * pos_unit
    vel = np.random.random(size=(3, 10)) * vel_unit
    t = np.linspace(0, 1, 10) * u.Myr

    if should_raise:
        with pytest.raises(u.UnitTypeError):
            Orbit(pos=pos, vel=vel, t=t)
    else:
        Orbit(pos=pos, vel=vel, t=t)
