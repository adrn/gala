import astropy.units as u
import numpy as np
import pytest

import gala.potential as gp
from gala._cconfig import GSL_ENABLED
from gala.units import galactic

# global pytest marker to skip tests if EXP is not enabled
pytestmark = pytest.mark.skipif(
    not GSL_ENABLED,
    reason="requires Gala compiled with GSL support",
)


@pytest.fixture
def potentials():
    time_knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) * u.Gyr
    m_knots = np.full(len(time_knots), 1e12)
    m_knots[-1] = 2e12

    pots = {}

    pots["base"] = gp.HernquistPotential(m_knots[0], 10.0, units=galactic)

    pots["timedep"] = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=m_knots,
        c=np.full(len(time_knots), 10.0),
        units=galactic,
        interpolation_method="linear",
    )

    pots["timedep-const"] = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=m_knots[0],
        c=np.full(len(time_knots), 10.0),
        units=galactic,
        interpolation_method="linear",
    )

    return pots


@pytest.fixture
def pos():
    return np.array([8.0, 7.0, 6.0])


@pytest.mark.parametrize("func_name", ["energy", "gradient", "density", "hessian"])
def test_timeinterp_same(func_name, pos, potentials):
    vals = {}
    for name, pot in potentials.items():
        vals[name] = getattr(pot, func_name)(pos, t=2.5 * u.Gyr)
    assert u.allclose(vals["base"], vals["timedep"])
    assert u.allclose(vals["base"], vals["timedep-const"])
    print(f"{func_name} evaluation: {vals['base']}, {vals['timedep']}")


@pytest.mark.parametrize("func_name", ["energy", "gradient", "density", "hessian"])
def test_timeinterp_diff(func_name, pos, potentials):
    vals = {}
    for name, pot in potentials.items():
        vals[name] = getattr(pot, func_name)(pos, t=4.5 * u.Gyr)
    assert u.allclose(vals["base"], vals["timedep-const"])
    assert not u.allclose(vals["base"], vals["timedep"])
    print(f"{func_name} evaluation: {vals['base']}, {vals['timedep']}")
