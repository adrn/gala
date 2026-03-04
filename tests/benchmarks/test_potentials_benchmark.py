import pathlib
import sys

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import EXP_ENABLED, GSL_ENABLED

import gala.potential as gp
from gala.units import SimulationUnitSystem, galactic

this_path = pathlib.Path(__file__).parent
potentials_test_path = (this_path / "../potential/potential").resolve()

# NOTE: this is a hack to allow importing from tests/potential/potential
sys.path.append(str(potentials_test_path))

from canonical_potentials import CANONICAL, NO_DENSITY  # noqa: E402


class BenchmarkPotentialBase:
    """Base class for potential benchmarks.

    Subclasses supply 'potential' either via a @pytest.fixture(scope="class")
    method or via class-level @pytest.mark.parametrize.

    Override the 'n_points' fixture in a subclass to restrict the point counts
    tested (e.g., for expensive potentials that are too slow at large n_points).
    """

    @pytest.fixture(params=[1, 10, 1_000, 10_000])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def rng(self):
        return np.random.default_rng(42)

    @pytest.mark.benchmark(max_rounds=16)
    def test_energy(self, potential, n_points, rng):
        potential.energy(rng.normal(0, 10, size=(potential.ndim, n_points)))

    @pytest.mark.benchmark(max_rounds=16)
    def test_gradient(self, potential, n_points, rng):
        potential.gradient(rng.normal(0, 10, size=(potential.ndim, n_points)))

    @pytest.mark.benchmark(max_rounds=16)
    def test_density(self, potential, n_points, rng):
        if type(potential).__name__ in NO_DENSITY:
            pytest.skip("density not implemented for this potential")  # type: ignore[call-arg]
        potential.density(rng.normal(0, 10, size=(potential.ndim, n_points)))


# ============================================================================
# Auto-generated benchmarks for all canonical potentials

_canonical_potentials = {
    name: pot for name, pot in CANONICAL.items() if pot is not None
}


@pytest.mark.parametrize(
    "potential",
    list(_canonical_potentials.values()),
    ids=list(_canonical_potentials.keys()),
)
class TestCanonicalPotentialsBenchmark(BenchmarkPotentialBase):
    pass


# ============================================================================
# Special (file-based potentials with expensive setup)


class TestCylSplineBenchmark(BenchmarkPotentialBase):
    @pytest.fixture(params=[1, 10, 1_000])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def potential(self):
        return gp.CylSplinePotential.from_file(
            potentials_test_path / "pot_disk_506151.pot", units=galactic
        )


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_density(BenchmarkPotentialBase):
    @pytest.fixture(scope="class")
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("density")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_potential(BenchmarkPotentialBase):
    @pytest.fixture(scope="class")
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("potential")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_mass(BenchmarkPotentialBase):
    @pytest.fixture(scope="class")
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("mass")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestTimeInterpolatedBenchmark(BenchmarkPotentialBase):
    @pytest.fixture(scope="class")
    def potential(self):
        time_knots = np.linspace(0, 5, 5) * u.Gyr
        m_vals = np.linspace(1e11, 2e11, 5) * u.Msun
        c_vals = np.linspace(0.26, 0.52, 5) * u.kpc
        return gp.TimeInterpolatedPotential(
            potential_cls=gp.HernquistPotential,
            time_knots=time_knots,
            m=m_vals,
            c=c_vals,
            units=galactic,
        )

    @pytest.mark.benchmark(max_rounds=16)
    def test_energy(self, potential, n_points, rng):
        potential.energy(rng.normal(0, 10, size=(potential.ndim, n_points)), t=2.5)

    @pytest.mark.benchmark(max_rounds=16)
    def test_gradient(self, potential, n_points, rng):
        potential.gradient(rng.normal(0, 10, size=(potential.ndim, n_points)), t=2.5)

    @pytest.mark.benchmark(max_rounds=16)
    def test_density(self, potential, n_points, rng):
        potential.density(rng.normal(0, 10, size=(potential.ndim, n_points)), t=2.5)


@pytest.mark.skipif(not EXP_ENABLED, reason="requires EXP")
class TestEXPStaticBenchmark(BenchmarkPotentialBase):
    @pytest.fixture(scope="class")
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
    @pytest.fixture(scope="class")
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
