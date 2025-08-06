"""
Test script for TimeInterpolatedPotential implementation.

This script tests the basic functionality of the TimeInterpolatedPotential
class to ensure it works correctly.
"""

import sys

import astropy.units as u
import numpy as np
from scipy.spatial.transform import Rotation as R

from gala.potential import KeplerPotential
from gala.units import galactic

# Try to import our new class
try:
    from gala.potential.potential.builtin.time_interpolated import (
        TimeInterpolatedPotential,
    )

    print("✓ Successfully imported TimeInterpolatedPotential")
except ImportError as e:
    print(f"✗ Failed to import TimeInterpolatedPotential: {e}")
    sys.exit(1)


def test_constant_parameters():
    """Test with constant parameters (should behave like regular potential)."""
    print("\n--- Testing constant parameters ---")

    # Create time knots
    times = np.linspace(0, 100, 11) * u.Myr

    # Constant mass
    mass = 1e10 * u.Msun

    try:
        # Create time-interpolated potential with constant mass
        pot_interp = TimeInterpolatedPotential(
            KeplerPotential, times, m=mass, units=galactic
        )

        # Create regular Kepler potential for comparison
        pot_regular = KeplerPotential(m=mass, units=galactic)

        # Test positions
        q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T

        # Evaluate at different times
        for t in [0 * u.Myr, 50 * u.Myr, 100 * u.Myr]:
            E_interp = pot_interp.energy(q, t=t)
            E_regular = pot_regular.energy(q)

            if np.allclose(E_interp.value, E_regular.value):
                print(f"✓ Energy matches at t={t}")
            else:
                print(f"✗ Energy mismatch at t={t}: {E_interp} vs {E_regular}")

    except Exception as e:
        print(f"✗ Error in constant parameters test: {e}")


def test_time_varying_parameters():
    """Test with time-varying parameters."""
    print("\n--- Testing time-varying parameters ---")

    try:
        # Create time knots
        times = np.linspace(0, 100, 11) * u.Myr

        # Mass growing linearly with time
        masses = np.linspace(1e10, 2e10, 11) * u.Msun

        # Create time-interpolated potential
        pot = TimeInterpolatedPotential(
            KeplerPotential, times, m=masses, units=galactic
        )

        # Test position
        q = np.array([[1.0, 0.0, 0.0]]).T

        # Check that mass varies with time
        E_start = pot.energy(q, t=0 * u.Myr)
        E_mid = pot.energy(q, t=50 * u.Myr)
        E_end = pot.energy(q, t=100 * u.Myr)

        print(f"Energy at t=0: {E_start}")
        print(f"Energy at t=50: {E_mid}")
        print(f"Energy at t=100: {E_end}")

        # Energy should become more negative as mass increases
        if E_start > E_mid > E_end:
            print("✓ Energy decreases with increasing mass")
        else:
            print("✗ Energy trend incorrect")

    except Exception as e:
        print(f"✗ Error in time-varying parameters test: {e}")


def test_rotation_interpolation():
    """Test rotation matrix interpolation."""
    print("\n--- Testing rotation interpolation ---")

    try:
        # Create time knots
        times = np.linspace(0, 100, 11) * u.Myr

        # Create rotation matrices (90 degree rotation over time)
        angles = np.linspace(0, np.pi / 2, 11)
        rotations = np.array(
            [R.from_rotvec([0, 0, angle]).as_matrix() for angle in angles]
        )

        # Create time-interpolated potential
        pot = TimeInterpolatedPotential(
            KeplerPotential, times, m=1e10 * u.Msun, R=rotations, units=galactic
        )

        # Test position along x-axis
        q = np.array([[1.0, 0.0, 0.0]]).T

        # At t=0, should be same as x-axis
        E_start = pot.energy(q, t=0 * u.Myr)

        # At t=100, position should be rotated by 90 degrees
        # So it should be like evaluating at [0, 1, 0] in the potential frame
        q_rotated = np.array([[0.0, 1.0, 0.0]]).T
        pot_regular = KeplerPotential(m=1e10 * u.Msun, units=galactic)
        E_expected = pot_regular.energy(q_rotated)
        E_end = pot.energy(q, t=100 * u.Myr)

        print(f"Energy at start: {E_start}")
        print(f"Energy at end: {E_end}")
        print(f"Expected energy: {E_expected}")

        if np.allclose(E_end.value, E_expected.value, rtol=1e-3):
            print("✓ Rotation interpolation works")
        else:
            print("✗ Rotation interpolation failed")

    except Exception as e:
        print(f"✗ Error in rotation interpolation test: {e}")


def test_bounds_checking():
    """Test that extrapolation raises errors."""
    print("\n--- Testing bounds checking ---")

    try:
        # Create time knots
        times = np.linspace(0, 100, 11) * u.Myr
        masses = np.linspace(1e10, 2e10, 11) * u.Msun

        # Create time-interpolated potential
        pot = TimeInterpolatedPotential(
            KeplerPotential, times, m=masses, units=galactic
        )

        # Test position
        q = np.array([[1.0, 0.0, 0.0]]).T

        # Try to evaluate outside bounds
        try:
            pot.energy(q, t=-10 * u.Myr)  # Before start
            print("✗ Should have raised error for t < t_min")
        except Exception as e:
            print("✓ Correctly raised error for t < t_min")

        try:
            pot.energy(q, t=110 * u.Myr)  # After end
            print("✗ Should have raised error for t > t_max")
        except Exception as e:
            print("✓ Correctly raised error for t > t_max")

    except Exception as e:
        print(f"✗ Error in bounds checking test: {e}")
