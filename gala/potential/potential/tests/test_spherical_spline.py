"""
Test the builtin CPotential classes

TODO:
- Test different valid interpolation methods
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G

import gala.potential as gp
from gala._cconfig import GSL_ENABLED

from ....units import galactic
from .helpers import PotentialTestBase

# global pytest marker to skip tests if GSL is not enabled
pytestmark = pytest.mark.skipif(
    not GSL_ENABLED,
    reason="requires Gala compiled with GSL support",
)


def _analytic_hernquist_potential():
    pot = gp.HernquistPotential(m=1e12 * u.Msun, c=18 * u.kpc, units=galactic)
    r_knots = np.geomspace(0.01, 1000, 128) * u.kpc
    return r_knots, pot


def _make_potential(kind):
    r_knots, pot = _analytic_hernquist_potential()
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
        interpolation_method="cspline",
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

    @pytest.mark.skip(reason="Not implemented for SphericalSpline potentials")
    def test_against_sympy(self):
        pass

    @pytest.mark.skip(reason="Not implemented for SphericalSpline potentials")
    def test_numerical_gradient_vs_gradient(self):
        pass

    @pytest.mark.parametrize("compare", ["energy", "density", "mass_enclosed"])
    def test_against_hernquist_potential(self, compare):
        r_knots, hern = _analytic_hernquist_potential()

        r_grid = (
            np.geomspace(r_knots.min().value * 10, r_knots.max().value / 10, 256)
            * r_knots.unit
        )
        xyz = np.stack((r_grid, np.zeros_like(r_grid), np.zeros_like(r_grid)), axis=0)
        spline_vals = getattr(self.potential, compare)(xyz)
        analytic_vals = getattr(hern, compare)(xyz)

        # Different spline types can accurately reproduce different quantities.
        # - Potential spline: should accurately reproduce energy, mass, and density
        #   (because they are computed from derivatives of the spline)
        # - Density spline: should accurately reproduce density (by construction).
        # - Mass spline: should accurately reproduce mass_enclosed (by construction) and
        #   density (derivative), but will only match analytic energy up to a
        #   multiplicative factor. And actually won't match mass_enclosed well because
        #   of the way that is computed in Gala, using the potential derivative.
        if (
            (self._spline_type == "potential" and compare == "energy")
            or (self._spline_type == "density" and compare == "density")
            or (self._spline_type == "mass" and compare == "mass_enclosed")
        ):
            # Remove edge effects near the edges of the knot grid -- internally, should
            # be robust (256 vs. 64 knots = 4 times as many points, so ignore 8 points
            # to ignore the last 2 knots on either end)
            assert u.allclose(spline_vals, analytic_vals, rtol=1e-3)

        # This involves the 2nd derivative of the spline, so looser tolerance
        elif self._spline_type == "potential" and compare == "density":
            assert u.allclose(spline_vals, analytic_vals, rtol=5e-2)

        # The next two cases need to be corrected for the fact that the spline has no
        # density inwards or outwards of the knot range
        elif self._spline_type == "density" and compare == "energy":
            M_inner = hern.mass_enclosed(r_knots.min() * [1.0, 0, 0])
            # The missing inner mass contributes -G*M_inner/r to the potential

            # We also need the constant offset from missing outer mass
            E_outer_offset = hern.energy(
                r_knots.max() * [1.0, 0, 0]
            ) - self.potential.energy(r_knots.max() * [1.0, 0, 0])

            # Total correction: position-dependent inner mass + constant outer offset
            E_correct = -G * M_inner / r_grid + E_outer_offset

            assert u.allclose(spline_vals + E_correct, analytic_vals, rtol=1e-3)

        elif self._spline_type == "density" and compare == "mass_enclosed":
            M_correct = hern.mass_enclosed(r_knots.min() * [1.0, 0, 0])
            assert u.allclose(spline_vals + M_correct, analytic_vals, rtol=1e-3)

        # Account for the fact that the spline has zero mass enclosed inside of the knot
        # range
        elif self._spline_type == "mass" and compare == "energy":
            M_inner = hern.mass_enclosed(r_knots.min() * [1.0, 0, 0])

            # Add constant offset from missing outer mass
            E_offset = hern.energy(r_knots.max() * [1.0, 0, 0]) - (
                self.potential.energy(r_knots.max() * [1.0, 0, 0])
            )

            E_correct = -G * M_inner / r_grid + E_offset

            assert u.allclose(spline_vals + E_correct, analytic_vals, rtol=1e-3)

        else:
            assert u.allclose(spline_vals, analytic_vals, rtol=1e-3)


class TestSphericalSpline_potential(SphericalSplineTestBase):
    _spline_type = "potential"


class TestSphericalSpline_density(SphericalSplineTestBase):
    _spline_type = "density"


class TestSphericalSpline_mass(SphericalSplineTestBase):
    _spline_type = "mass"
