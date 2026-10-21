import pathlib
import sys

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import EXP_ENABLED, GSL_ENABLED

import gala.dynamics as gd
import gala.potential as gp
from gala.units import SimulationUnitSystem, galactic

this_path = pathlib.Path(__file__).parent
potentials_test_path = (this_path / "../potential/potential").resolve()

# NOTE: this is a hack to allow importing from tests/potential/potential
sys.path.append(str(potentials_test_path))

from canonical_potentials import CANONICAL, NO_DENSITY


class BenchmarkPotentialBase:
    """Base class for potential benchmarks.

    Subclasses supply 'potential' either via a @pytest.fixture(scope="class")
    method or via class-level @pytest.mark.parametrize.

    Override the 'n_points' fixture in a subclass to restrict the point counts
    tested (e.g., for expensive potentials that are too slow at large n_points).
    """

    @pytest.fixture(params=[1, 1_000, 10_000])
    def n_points(self, request):
        return request.param

    @pytest.fixture
    def q(self, potential, n_points):
        # A fixture, so that generating the positions is not timed
        return np.random.default_rng(42).normal(0, 10, size=(potential.ndim, n_points))

    @pytest.mark.benchmark
    def test_energy(self, potential, q):
        potential.energy(q)

    @pytest.mark.benchmark
    def test_gradient(self, potential, q):
        potential.gradient(q)

    @pytest.mark.benchmark
    def test_density(self, potential, q):
        if type(potential).__name__ in NO_DENSITY:
            pytest.skip("density not implemented for this potential")  # type: ignore[call-arg]
        potential.density(q)


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


@pytest.fixture
def positions(n_points):
    return np.random.default_rng(42).normal(0, 10, size=(3, n_points))


@pytest.mark.benchmark
@pytest.mark.parametrize("n_points", [1, 1_000, 10_000])
def test_hessian(positions):
    # Just one representative potential
    CANONICAL["HernquistPotential"].hessian(positions)


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
    @pytest.fixture(params=[1, 10, 1_000])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("density")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_potential(BenchmarkPotentialBase):
    @pytest.fixture(params=[1, 10, 1_000])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def potential(self):
        from test_spherical_spline import _make_potential

        return _make_potential("potential")


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
class TestSphericalSplineBenchmark_mass(BenchmarkPotentialBase):
    @pytest.fixture(params=[1, 10, 1_000])
    def n_points(self, request):
        return request.param

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

    @pytest.mark.benchmark
    def test_energy(self, potential, q):
        potential.energy(q, t=2.5)

    @pytest.mark.benchmark
    def test_gradient(self, potential, q):
        potential.gradient(q, t=2.5)

    @pytest.mark.benchmark
    def test_density(self, potential, q):
        potential.density(q, t=2.5)


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


# ============================================================================
# Input types other than plain arrays, which are converted before evaluation


@pytest.fixture
def positions_kpc(positions):
    return positions * u.kpc


@pytest.mark.benchmark
@pytest.mark.parametrize("n_points", [1, 10_000])
def test_energy_quantity(positions_kpc):
    CANONICAL["HernquistPotential"].energy(positions_kpc)


@pytest.fixture(scope="module")
def orbit():
    # 1,000 orbits x 1,000 times: large enough that copies of the orbit cost more
    # than the fixed overhead of a call
    rng = np.random.default_rng(42)
    w0 = gd.PhaseSpacePosition(
        pos=rng.normal(0, 10, size=(3, 1000)) * u.kpc,
        vel=rng.normal(0, 100, size=(3, 1000)) * u.km / u.s,
    )
    return CANONICAL["HernquistPotential"].integrate_orbit(w0, dt=1.0, n_steps=999)


@pytest.mark.benchmark
def test_energy_orbit(orbit):
    CANONICAL["HernquistPotential"].energy(orbit)


@pytest.mark.benchmark
def test_orbit_energy_method(orbit):
    orbit.energy()
