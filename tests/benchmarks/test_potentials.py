import functools
import pathlib
import sys

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import EXP_ENABLED, GSL_ENABLED

import gala.potential as gp
from gala.units import SimulationUnitSystem

this_path = pathlib.Path(__file__).parent
potentials_test_path = (this_path / "../potential/potential").resolve()

# NOTE: this is a hack to allow importing from tests/potential/potential
sys.path.append(str(potentials_test_path))


@pytest.mark.parametrize("n_points", [1, 10, 1_000, 100_000])
class BenchmarkPotentialBase:
    @pytest.fixture(scope="class")
    def rng(self):
        return np.random.default_rng(42)

    def sample_xyz(self, n_points, rng):
        return rng.normal(0, 10, size=(3, n_points))

    def sample_vxyz(self, n_points, rng):
        return rng.normal(0, 100, size=(3, n_points))

    @pytest.mark.benchmark(max_rounds=16)
    def test_evaluate_potential(self, n_points, rng):
        self.potential.energy(self.sample_xyz(n_points, rng))

    @pytest.mark.benchmark(max_rounds=16)
    def test_evaluate_gradient(self, n_points, rng):
        self.potential.gradient(self.sample_xyz(n_points, rng))

    @pytest.mark.benchmark(max_rounds=16)
    def test_evaluate_density(self, n_points, rng):
        self.potential.density(self.sample_xyz(n_points, rng))


# ============================================================================
# Spherical


class TestHernquistBenchmark(BenchmarkPotentialBase):
    potential = gp.HernquistPotential(m=1e11, c=5, units="galactic")


# ============================================================================
# Special


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_density(BenchmarkPotentialBase):
    @functools.cached_property
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("density")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_potential(BenchmarkPotentialBase):
    @functools.cached_property
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("potential")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_mass(BenchmarkPotentialBase):
    @functools.cached_property
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("mass")


@pytest.mark.skipif(not EXP_ENABLED, reason="requires EXP")
class TestEXPStaticBenchmark(BenchmarkPotentialBase):
    @functools.cached_property
    def potential(self):
        """Note: See tests/potential/potential/test_exp.py"""
        exp_units = SimulationUnitSystem(
            mass=1.25234e11 * u.Msun, length=3.845 * u.kpc, G=1
        )
        EXP_CONFIG_FILE = str(potentials_test_path / "EXP-Hernquist-basis.yml")
        EXP_SINGLE_COEF_FILE = str(
            potentials_test_path / "EXP-Hernquist-single-coefs.hdf5"
        )
        return gp.EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=EXP_SINGLE_COEF_FILE,
            units=exp_units,
        )


@pytest.mark.skipif(not EXP_ENABLED, reason="requires EXP")
class TestEXPTimeInterpBenchmark(BenchmarkPotentialBase):
    @functools.cached_property
    def potential(self):
        """Note: See tests/potential/potential/test_exp.py"""
        exp_units = SimulationUnitSystem(
            mass=1.25234e11 * u.Msun, length=3.845 * u.kpc, G=1
        )
        EXP_CONFIG_FILE = str(potentials_test_path / "EXP-Hernquist-basis.yml")
        EXP_SINGLE_COEF_FILE = str(
            potentials_test_path / "EXP-Hernquist-multi-coefs.hdf5"
        )
        return gp.EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=EXP_SINGLE_COEF_FILE,
            units=exp_units,
        )
