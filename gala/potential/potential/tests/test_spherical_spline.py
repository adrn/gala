"""
Test the builtin CPotential classes
"""

import astropy.units as u
import numpy as np
import pytest

import gala.potential as gp
from gala._cconfig import GSL_ENABLED

from ....units import galactic
from .helpers import PotentialTestBase

# global pytest marker to skip tests if GSL is not enabled
pytestmark = pytest.mark.skipif(
    not GSL_ENABLED,
    reason="requires Gala compiled with GSL support",
)


def _make_potential(kind):
    pot = gp.HernquistPotential(m=1e12 * u.Msun, c=18 * u.kpc, units=galactic)

    r_knots = np.geomspace(0.1, 100, 64) * u.kpc
    r_xyz = np.stack((r_knots, np.zeros_like(r_knots), np.zeros_like(r_knots)), axis=0)

    if kind == "potential":
        vals = pot.energy(r_xyz)
    elif kind == "density":
        vals = pot.density(r_xyz)
    elif kind == "mass":
        vals = pot.mass_enclosed(r_xyz)
    else:
        raise ValueError("Invalid kind")

    return gp.SphericalSplinePotential(
        r_knots=r_knots,
        spline_values=vals,
        spline_value_type=kind,
        interpolation_method="linear",
        units=galactic,
    )


class SphericalSplineTestBase(PotentialTestBase):
    w0 = [8.0, 0.0, 0.0, 0.0, 0.1, 0.1]
    # atol = 1e-3
    sympy_density = False
    check_finite_at_origin = False

    def setup_method(self):
        self.potential = _make_potential(self._spline_type)
        super().setup_method()

    @pytest.mark.skip(reason="Not implemented for rotated potentials")
    def test_against_sympy(self):
        pass


class TestSphericalSpline_potential(SphericalSplineTestBase):
    _spline_type = "potential"


class TestSphericalSpline_density(SphericalSplineTestBase):
    _spline_type = "density"


class TestSphericalSpline_mass(SphericalSplineTestBase):
    _spline_type = "mass"
