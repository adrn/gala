"""
Test script for TimeInterpolatedPotential implementation.

This script tests the basic functionality of the TimeInterpolatedPotential
class to ensure it works correctly.
"""

import astropy.units as u
import numpy as np

import gala.potential as gp
from gala.units import galactic


def test_simple():
    # Create a simple Hernquist potential first to check it works normally
    hernquist = gp.HernquistPotential(m=1e12, c=10.0, units=galactic)
    pos = np.array([8.0, 0.0, 0.0])

    print("Testing normal Hernquist potential:")
    energy = hernquist.energy(pos)
    print(f"Normal Hernquist energy: {energy}")

    # Now test time-interpolated with truly constant single-value parameters
    time_knots = np.array([0.0, 1.0])
    print("\nTesting time-interpolated with single values:")
    time_pot1 = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=[1e12],  # Single value
        c=[10.0],  # Single value
        units=galactic,
    )
    print("✓ Single-value TimeInterpolatedPotential created")

    energy1 = time_pot1.energy(pos, t=0.5)
    print(f"Single-value energy: {energy1}")

    # Now test with explicit array of same values
    print("\nTesting time-interpolated with array of same values:")
    time_pot2 = gp.TimeInterpolatedPotential(
        potential_cls=gp.HernquistPotential,
        time_knots=time_knots,
        m=np.array([1e12, 1e12]),  # Array of same values
        c=np.array([10.0, 10.0]),  # Array of same values
        units=galactic,
    )
    print("✓ Array-value TimeInterpolatedPotential created")

    energy2 = time_pot2.energy(pos, t=0.5)
    print(f"Array-value energy: {energy2}")


def test_constant_parameters():
    """Test with constant parameters (should behave like regular potential)."""
    times = np.linspace(0, 100, 11) * u.Myr
    mass = 1e10 * u.Msun

    pot_interp = gp.TimeInterpolatedPotential(
        gp.KeplerPotential, time_knots=times, m=mass, units=galactic
    )
    pot_regular = gp.KeplerPotential(m=mass, units=galactic)

    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T
    for t in [0 * u.Myr, 50 * u.Myr, 100 * u.Myr]:
        E_interp = pot_interp.energy(q, t=t)
        E_regular = pot_regular.energy(q)
        assert u.allclose(E_interp, E_regular)


# def test_time_varying_parameters():
#     """Test with time-varying parameters."""

#     # Create time knots
#     times = np.linspace(0, 100, 11) * u.Myr

#     # Mass growing linearly with time
#     masses = np.linspace(1e10, 2e10, 11) * u.Msun

#     # Create time-interpolated potential
#     pot = TimeInterpolatedPotential(KeplerPotential, times, m=masses, units=galactic)

#     # Test position
#     q = np.array([[1.0, 0.0, 0.0]]).T

#     # Check that mass varies with time
#     E_start = pot.energy(q, t=0 * u.Myr)
#     E_mid = pot.energy(q, t=50 * u.Myr)
#     E_end = pot.energy(q, t=100 * u.Myr)

#     print(f"Energy at t=0: {E_start}")
#     print(f"Energy at t=50: {E_mid}")
#     print(f"Energy at t=100: {E_end}")


# def test_rotation_interpolation():
#     """Test rotation matrix interpolation."""
#     print("\n--- Testing rotation interpolation ---")

#     try:
#         # Create time knots
#         times = np.linspace(0, 100, 11) * u.Myr

#         # Create rotation matrices (90 degree rotation over time)
#         angles = np.linspace(0, np.pi / 2, 11)
#         rotations = np.array(
#             [R.from_rotvec([0, 0, angle]).as_matrix() for angle in angles]
#         )

#         # Create time-interpolated potential
#         pot = TimeInterpolatedPotential(
#             KeplerPotential, times, m=1e10 * u.Msun, R=rotations, units=galactic
#         )

#         # Test position along x-axis
#         q = np.array([[1.0, 0.0, 0.0]]).T

#         # At t=0, should be same as x-axis
#         E_start = pot.energy(q, t=0 * u.Myr)

#         # At t=100, position should be rotated by 90 degrees
#         # So it should be like evaluating at [0, 1, 0] in the potential frame
#         q_rotated = np.array([[0.0, 1.0, 0.0]]).T
#         pot_regular = KeplerPotential(m=1e10 * u.Msun, units=galactic)
#         E_expected = pot_regular.energy(q_rotated)
#         E_end = pot.energy(q, t=100 * u.Myr)

#         print(f"Energy at start: {E_start}")
#         print(f"Energy at end: {E_end}")
#         print(f"Expected energy: {E_expected}")

#         if np.allclose(E_end.value, E_expected.value, rtol=1e-3):
#             print("✓ Rotation interpolation works")
#         else:
#             print("✗ Rotation interpolation failed")

#     except Exception as e:
#         print(f"✗ Error in rotation interpolation test: {e}")


# def test_bounds_checking():
#     """Test that extrapolation raises errors."""
#     print("\n--- Testing bounds checking ---")

#     try:
#         # Create time knots
#         times = np.linspace(0, 100, 11) * u.Myr
#         masses = np.linspace(1e10, 2e10, 11) * u.Msun

#         # Create time-interpolated potential
#         pot = TimeInterpolatedPotential(
#             KeplerPotential, times, m=masses, units=galactic
#         )

#         # Test position
#         q = np.array([[1.0, 0.0, 0.0]]).T

#         # Try to evaluate outside bounds
#         try:
#             pot.energy(q, t=-10 * u.Myr)  # Before start
#             print("✗ Should have raised error for t < t_min")
#         except Exception as e:
#             print("✓ Correctly raised error for t < t_min")

#         try:
#             pot.energy(q, t=110 * u.Myr)  # After end
#             print("✗ Should have raised error for t > t_max")
#         except Exception as e:
#             print("✓ Correctly raised error for t > t_max")

#     except Exception as e:
#         print(f"✗ Error in bounds checking test: {e}")
