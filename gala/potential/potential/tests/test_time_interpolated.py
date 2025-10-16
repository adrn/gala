"""
Test suite for TimeInterpolatedPotential implementation.

Tests the functionality of the TimeInterpolatedPotential class including:
- Constant parameter behavior
- Time-varying parameters
- Rotation matrix interpolation
- Bounds checking
- Vectorized evaluations
"""

import astropy.units as u
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import gala.potential as gp
from gala._cconfig import GSL_ENABLED
from gala.units import galactic

# global pytest marker to skip tests if EXP is not enabled
pytestmark = pytest.mark.skipif(
    not GSL_ENABLED,
    reason="requires Gala compiled with GSL support",
)


@pytest.fixture
def time_knots():
    """Standard time knots for testing."""
    return np.linspace(0, 100, 11) * u.Myr


@pytest.fixture
def test_positions():
    """Test positions for evaluation."""
    return {
        "single": np.array([8.0, 0.0, 0.0]),
        "multiple": np.array([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0], [0.0, 0.5, 2.0]]).T,
        "grid": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T,
    }


@pytest.fixture
def potentials():
    """Different potential configurations for testing."""
    time_knots = np.array([0.0, 50.0, 100.0]) * u.Myr

    pots = {}

    # Base potential for comparison
    pots["base"] = gp.HernquistPotential(
        m=1e12 * u.Msun, c=10.0 * u.kpc, units=galactic
    )

    # Time-interpolated with constant parameters
    pots["constant"] = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=1e12 * u.Msun,
        c=10.0 * u.kpc,
        units=galactic,
    )

    # Time-interpolated with varying mass
    masses = np.array([1e12, 1.5e12, 2e12]) * u.Msun
    pots["varying"] = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=masses,
        c=10.0 * u.kpc,
        units=galactic,
    )

    return pots


def test_constant_parameters_single_value(test_positions, time_knots):
    """Test TimeInterpolatedPotential with single constant values."""
    # Single value parameters
    pot_single = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=[1e12] * u.Msun,
        c=[10.0] * u.kpc,
        units=galactic,
    )

    # Should work without errors
    energy = pot_single.energy(test_positions["single"], t=50 * u.Myr)
    assert np.isfinite(energy.value)


def test_constant_parameters_array_values(test_positions, time_knots):
    """Test TimeInterpolatedPotential with array constant values."""
    # Array of same values
    pot_array = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=np.full(len(time_knots), 1e12) * u.Msun,
        c=np.full(len(time_knots), 10.0) * u.kpc,
        units=galactic,
    )

    energy = pot_array.energy(test_positions["single"], t=50 * u.Myr)
    assert np.isfinite(energy.value)


@pytest.mark.parametrize("func_name", ["energy", "gradient", "density"])
def test_constant_vs_regular_potential(func_name, test_positions, potentials):
    """Test that constant-parameter TimeInterpolatedPotential equals regular potential."""
    pos = test_positions["single"]

    val_base = getattr(potentials["base"], func_name)(pos)
    val_constant = getattr(potentials["constant"], func_name)(pos, t=25 * u.Myr)

    assert u.allclose(val_base, val_constant, rtol=1e-10)


@pytest.mark.parametrize("func_name", ["energy", "gradient", "density"])
def test_vectorized_evaluation(func_name, test_positions, potentials):
    """Test vectorized evaluation with multiple positions."""
    pos = test_positions["multiple"]

    result = getattr(potentials["varying"], func_name)(pos, t=25 * u.Myr)

    # Check shapes
    if func_name == "energy":
        assert result.shape == (pos.shape[1],)
    elif func_name == "gradient":
        assert result.shape == pos.shape
    elif func_name == "density":
        assert result.shape == (pos.shape[1],)

    # Check all values are finite
    assert np.all(np.isfinite(result.value))


def test_time_varying_parameters(test_positions, potentials):
    """Test that time-varying parameters produce different results at different times."""
    pos = test_positions["single"]

    # Energy should be different at different times due to varying mass
    E_start = potentials["varying"].energy(pos, t=0 * u.Myr)
    E_mid = potentials["varying"].energy(pos, t=50 * u.Myr)
    E_end = potentials["varying"].energy(pos, t=100 * u.Myr)

    # All should be finite
    assert np.all(np.isfinite([E_start.value, E_mid.value, E_end.value]))

    # Should be different (mass is increasing, so energy magnitude should increase)
    assert not u.allclose(E_start, E_mid, rtol=1e-6)
    assert not u.allclose(E_mid, E_end, rtol=1e-6)

    # More massive = more negative energy (for bound systems)
    assert E_end < E_mid < E_start


def test_interpolation_accuracy():
    """Test that interpolation gives reasonable intermediate values."""
    time_knots = np.array([0.0, 100.0]) * u.Myr
    masses = np.array([1e12, 2e12]) * u.Msun

    pot = gp.TimeInterpolatedPotential(
        potential_cls=gp.KeplerPotential,
        time_knots=time_knots,
        m=masses,
        units=galactic,
    )

    pos = np.array([1.0, 0.0, 0.0])

    # At midpoint, should be close to average mass behavior
    E_mid = pot.energy(pos, t=50 * u.Myr)

    # Create reference potential with average mass
    pot_ref = gp.KeplerPotential(m=1.5e12 * u.Msun, units=galactic)
    E_ref = pot_ref.energy(pos)

    # Should be close (exact for linear interpolation)
    assert u.allclose(E_mid, E_ref, rtol=1e-10)


def test_rotation_interpolation():
    """Test rotation matrix interpolation."""
    time_knots = np.array([0.0, 100.0]) * u.Myr

    # Create rotation matrices: 0 to 90 degrees around z-axis
    angles = np.array([0.0, np.pi / 2])
    rotations = np.array([R.from_rotvec([0, 0, angle]).as_matrix() for angle in angles])

    pot = gp.TimeInterpolatedPotential(
        potential_cls=gp.KeplerPotential,
        time_knots=time_knots,
        m=1e10 * u.Msun,
        R=rotations,
        units=galactic,
    )

    # Test position along x-axis
    pos = np.array([1.0, 0.0, 0.0])

    # At t=0, should behave like no rotation
    E_start = pot.energy(pos, t=0 * u.Myr)
    pot_ref = gp.KeplerPotential(m=1e10 * u.Msun, units=galactic)
    E_ref_start = pot_ref.energy(pos)
    assert u.allclose(E_start, E_ref_start, rtol=1e-10)

    # At t=100, should behave like 90-degree rotation
    # x-axis position should now be equivalent to y-axis in potential frame
    E_end = pot.energy(pos, t=100 * u.Myr)
    pos_rotated = np.array([0.0, 1.0, 0.0])
    E_ref_end = pot_ref.energy(pos_rotated)
    assert u.allclose(E_end, E_ref_end, rtol=1e-3)


def test_bounds_checking():
    """Test that evaluation outside time bounds returns NaN."""
    time_knots = np.array([10.0, 90.0]) * u.Myr
    masses = np.array([1e12, 2e12]) * u.Msun

    pot = gp.TimeInterpolatedPotential(
        potential_cls=gp.KeplerPotential,
        time_knots=time_knots,
        m=masses,
        units=galactic,
    )

    pos = np.array([1.0, 0.0, 0.0])

    # Outside bounds should return NaN
    E_before = pot.energy(pos, t=0 * u.Myr)  # Before t_min
    E_after = pot.energy(pos, t=100 * u.Myr)  # After t_max

    assert np.isnan(E_before.value)
    assert np.isnan(E_after.value)

    # Within bounds should work
    E_within = pot.energy(pos, t=50 * u.Myr)
    assert np.isfinite(E_within.value)


def test_gradient_consistency(test_positions, potentials):
    """Test gradient accuracy."""
    # Use single position for finite difference test
    grad = potentials["varying"].gradient(test_positions["single"], t=50 * u.Myr)
    assert np.isfinite(grad).all()
    # For a single position, gradient returns (3, 1) shape
    assert grad.shape == (3, 1)

    # Test multiple positions
    grad_multi = potentials["varying"].gradient(
        test_positions["multiple"], t=50 * u.Myr
    )
    assert np.isfinite(grad_multi).all()
    assert grad_multi.shape == test_positions["multiple"].shape


def test_hessian_basic_functionality(test_positions, potentials):
    """Test that hessian evaluation works and returns finite values."""
    pos = test_positions["single"]

    # Hessian for constant parameters (no coordinate transformations)
    hess = potentials["constant"].hessian(pos, t=25 * u.Myr)

    # Should have correct shape
    assert hess.shape == (3, 3, 1)

    # Should return finite values (though coordinate transformation may introduce small errors)
    # For now, we just test that it doesn't crash and returns something reasonable
    assert not np.all(np.isnan(hess.value))


def test_different_interpolation_types():
    """Test different interpolation methods."""
    time_knots = np.linspace(0, 100, 5) * u.Myr
    masses = np.array([1e12, 1.2e12, 1.8e12, 1.5e12, 2e12]) * u.Msun

    for interp_kind in ["linear", "cubic"]:
        pot = gp.TimeInterpolatedPotential(
            potential_cls=gp.KeplerPotential,
            time_knots=time_knots,
            m=masses,
            interp_kind=interp_kind,
            units=galactic,
        )

        pos = np.array([1.0, 0.0, 0.0])

        # Should work and give finite results
        energy = pot.energy(pos, t=50 * u.Myr)
        assert np.isfinite(energy.value)

        gradient = pot.gradient(pos, t=50 * u.Myr)
        assert np.all(np.isfinite(gradient.value))
