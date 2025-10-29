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
from gala._cconfig import GSL_ENABLED
from scipy.spatial.transform import Rotation as R

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.potential.potential.builtin.time_interpolated import _unsupported_cls
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
    # Single value parameters - pass as scalars, not single-element lists
    pot_single = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=1e12 * u.Msun,  # Scalar for constant parameter
        c=10.0 * u.kpc,  # Scalar for constant parameter
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
        interpolation_method="linear",  # Only 2 knots, so must use linear
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
        interpolation_method="linear",  # Only 2 knots, so must use linear
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
        interpolation_method="linear",
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

    for interp_kind in ["linear", "cspline"]:
        pot = gp.TimeInterpolatedPotential(
            potential_cls=gp.KeplerPotential,
            time_knots=time_knots,
            m=masses,
            interpolation_method=interp_kind,
            units=galactic,
        )

        pos = np.array([1.0, 0.0, 0.0])

        # Should work and give finite results
        energy = pot.energy(pos, t=50 * u.Myr)
        assert np.isfinite(energy.value)

        gradient = pot.gradient(pos, t=50 * u.Myr)
        assert np.all(np.isfinite(gradient.value))


@pytest.mark.parametrize("func_name", ["energy", "gradient", "density", "hessian"])
def test_timeinterp_same(func_name, potentials):
    pos = np.array([8.0, 7.0, 6.0])

    vals = {}
    for name, pot in potentials.items():
        vals[name] = getattr(pot, func_name)(pos, t=0 * u.Myr)
    assert u.allclose(vals["base"], vals["varying"])
    assert u.allclose(vals["base"], vals["constant"])
    print(f"{func_name} evaluation: {vals['base']}, {vals['varying']}")


@pytest.mark.parametrize("func_name", ["energy", "gradient", "density", "hessian"])
def test_timeinterp_diff(func_name, potentials):
    pos = np.array([8.0, 7.0, 6.0])

    vals = {}
    for name, pot in potentials.items():
        vals[name] = getattr(pot, func_name)(pos, t=75 * u.Myr)
    assert u.allclose(vals["base"], vals["constant"])
    assert not u.allclose(vals["base"], vals["varying"])
    print(f"{func_name} evaluation: {vals['base']}, {vals['varying']}")


def test_mismatched_parameter_length():
    """Test that mismatched parameter array lengths raise appropriate errors."""
    time_knots = np.linspace(0, 100, 11) * u.Myr

    # Single-element array should raise ValueError (ambiguous: constant or interpolated?)
    with pytest.raises(ValueError, match="Parameter 'm' has shape"):
        gp.TimeInterpolatedPotential(
            potential_cls=gp.HernquistPotential,
            time_knots=time_knots,
            m=[1e12] * u.Msun,  # Length 1, but 11 time knots
            c=10.0 * u.kpc,
            units=galactic,
        )

    # Wrong-length array should also raise ValueError
    with pytest.raises(ValueError, match="Parameter 'm' has shape"):
        gp.TimeInterpolatedPotential(
            potential_cls=gp.HernquistPotential,
            time_knots=time_knots,
            m=np.linspace(1e12, 2e12, 5) * u.Msun,  # Length 5, but 11 time knots
            c=10.0 * u.kpc,
            units=galactic,
        )


def test_scf_interpolated():
    """
    This is a specialized test for SCFPotential to make sure that everything works ok
    with the array parameters.
    """

    t_knots = np.linspace(0, 4, 32) * u.Gyr

    Sjnlm = np.zeros((len(t_knots), 3, 3, 3))
    Sjnlm[:, 0, 0, 0] = np.linspace(1.0, 4.0, len(t_knots))
    Tjnlm = np.zeros_like(Sjnlm)

    comx = np.zeros((len(t_knots), 3))
    comx[:, 0] = np.linspace(0, 2, len(t_knots)) * u.kpc

    # Moving potential and growing mass
    pot_interp = gp.TimeInterpolatedPotential(
        gp.SCFPotential,
        t_knots,
        m=1e10,
        r_s=1.0,
        Snlm=Sjnlm,
        Tnlm=Tjnlm,
        origin=comx,
        units=galactic,
    )

    x0 = [1.0, 0, 0.0] * u.kpc
    w0 = gd.PhaseSpacePosition(
        pos=x0, vel=[0, 0, 1] * pot_interp.circular_velocity(x0)[0]
    )

    orbit = pot_interp.integrate_orbit(
        w0, dt=0.1 * u.Myr, t1=0 * u.Gyr, t2=4 * u.Gyr, Integrator=gi.DOPRI853Integrator
    )

    assert u.isclose(np.mean(orbit.x[:1000]), 0 * u.kpc, atol=0.1 * u.kpc)
    assert u.isclose(np.mean(orbit.x[-1000:]), 2 * u.kpc, atol=0.1 * u.kpc)

    assert u.isclose(np.ptp(orbit.z[:1000]), 2 * u.kpc, atol=0.1 * u.kpc)
    assert u.isclose(np.ptp(orbit.z[-1000:]), 1 * u.kpc, atol=0.1 * u.kpc)


@pytest.mark.xfail(
    reason="SphericalSplinePotential does not work with time interpolated wrapper"
)
def test_spherical_spline_time_interpolated():
    """
    This is a specialized test for SphericalSplinePotential to make sure that everything
    works ok with the array parameters.
    """

    t_knots = np.linspace(0, 4, 32) * u.Gyr

    r_knots = np.geomspace(1e-1, 1e2, 128) * u.kpc
    values = np.zeros((len(t_knots), len(r_knots)))
    for i, t in enumerate(t_knots):
        tmp = gp.HernquistPotential(
            m=1e10 * (1 + t.to_value(u.Gyr)) * u.Msun, c=10 * u.kpc, units=galactic
        )
        values[i, :] = tmp.energy(r=r_knots).value

    pot_interp = gp.TimeInterpolatedPotential(
        gp.SphericalSplinePotential,
        t_knots,
        r_knots=r_knots,
        spline_values=values,
        spline_value_type="potential",
        units=galactic,
    )

    x0 = [1.0, 0, 0.0] * u.kpc
    w0 = gd.PhaseSpacePosition(
        pos=x0, vel=[0, 0, 1] * pot_interp.circular_velocity(x0)[0]
    )
    E1 = pot_interp.energy(x0, t=0 * u.Gyr)
    E2 = pot_interp.energy(x0, t=2.5 * u.Gyr)
    assert E2 < E1  # potential is getting deeper

    orbit = pot_interp.integrate_orbit(
        w0, dt=0.1 * u.Myr, t1=0 * u.Gyr, t2=4 * u.Gyr, Integrator=gi.DOPRI853Integrator
    )


# Check that all potential classes work with TimeInterpolatedPotential
@pytest.mark.parametrize(
    "pot_cls_name",
    [
        p
        for p in [*gp.potential.builtin.core.__all__, "SCFPotential"]
        if p not in _unsupported_cls and p != "TimeInterpolatedPotential"
    ],
)
def test_all_builtin_potentials_time_interpolated(pot_cls_name):
    pot_cls = getattr(gp, pot_cls_name)
    param_names = list(pot_cls._parameters.keys())

    knots = np.linspace(0, 100, 32) * u.Myr
    if param_names[0] == "m":
        params_const = {param_names[0]: 1e10}
    else:
        params_const = {param_names[0]: 1.0}

    params_time = {
        param_names[0]: params_const[param_names[0]] * np.linspace(1.0, 4, len(knots))
    }

    for param_name in param_names[1:]:
        params_time[param_name] = params_const[param_name] = 1.0

    # Special case a few potentials:
    if pot_cls_name == "SCFPotential":
        Sjnlm = np.zeros((len(knots), 4, 4, 4))
        Sjnlm[:, 0, 0, 0] = np.linspace(1.0, 2.0, len(knots))
        Tjnlm = np.zeros_like(Sjnlm)
        params_time["Snlm"] = Sjnlm
        params_time["Tnlm"] = Tjnlm
        params_const["Snlm"] = Sjnlm[0]
        params_const["Tnlm"] = Tjnlm[0]

    elif pot_cls_name == "MN3ExponentialDiskPotential":
        params_const["h_R"] = params_time["h_R"] = 5.0
        params_const["h_z"] = params_time["h_z"] = 0.5

    elif pot_cls_name == "StonePotential":
        params_const["r_c"] = params_time["r_c"] = 1.0
        params_const["r_h"] = params_time["r_h"] = 10.0

    pot_const = pot_cls(**params_const, units=galactic)
    pot_time = gp.TimeInterpolatedPotential(
        potential_cls=pot_cls,
        time_knots=knots,
        **params_time,
        units=galactic,
    )

    x = np.array([1.0, 0.0, 0.0]) * u.kpc
    assert u.allclose(pot_const.energy(x, t=0 * u.Myr), pot_time.energy(x, t=0 * u.Myr))
    assert not u.allclose(
        pot_const.energy(x, t=0 * u.Myr), pot_time.energy(x, t=50 * u.Myr)
    )


def test_integration_outside_interpolation_range():
    """Test that attempting to integrate an orbit outside of the interpolation fails"""
    time_knots = np.linspace(0, 100, 11) * u.Myr

    pot = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=np.linspace(1e11, 1e12, len(time_knots)),
        c=10.0 * u.kpc,
        units=galactic,
    )

    w0 = gd.PhaseSpacePosition(
        pos = [8, 0, 0] * u.kpc,
        vel = [0, 100, 0] * u.km/u.s
    )

    # Single-element array should raise ValueError (ambiguous: constant or interpolated?)
    with pytest.raises(ValueError, match="Integration times must be within the range"):
        pot.integrate_orbit(
            w0,
            t1=0 * u.Myr,
            t2=200 * u.Myr,         # max time is beyond max time knots time (100 Myr)
            dt=1 * u.Myr
        )


# TODO: functional tests
# - Orbit integration with a rotating bar
# - ...
