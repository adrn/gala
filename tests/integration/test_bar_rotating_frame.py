"""
Integration test for time-dependent barred potential vs. bar with a rotating frame.

This test validates that orbits integrated in a rotating frame match orbits integrated
in an inertial frame with a time-dependent rotating bar potential when transformed to
the rotating frame.
"""

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import GSL_ENABLED
from scipy.spatial.transform import Rotation

import gala.dynamics as gd
import gala.potential as gp

pytestmark = pytest.mark.skipif(
    not GSL_ENABLED,
    reason="requires Gala compiled with GSL support",
)


class TestBarRotatingFrameIntegration:
    """Test integration of orbits in rotating bar potentials."""

    @pytest.fixture
    def setup_potentials(self):
        """Set up the barred Milky Way potential and rotating frame."""
        # Set up rotation parameters
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            Omega = 30 * u.km / u.s / u.kpc
            Omega = Omega.to(u.rad / u.Gyr)

        dt = 2 * np.pi * u.rad / Omega / 200  # 200 steps per rotation period

        # Create time-dependent rotation matrices
        time_knots = np.arange(0, 5, dt.to(u.Gyr).value) * u.Gyr
        bar_angle = (-Omega * time_knots).to_value(u.rad)
        Rs = Rotation.from_euler("z", bar_angle).as_matrix()

        # Base Milky Way potential
        mw = gp.MilkyWayPotential(version="latest")

        # Time-dependent barred potential (inertial frame)
        bar_mw = gp.CCompositePotential()
        bar_mw["bar"] = gp.TimeInterpolatedPotential(
            gp.LongMuraliBarPotential,
            time_knots=time_knots,
            m=1e10 * u.Msun,
            a=4 * u.kpc,
            b=0.8 * u.kpc,
            c=0.25 * u.kpc,
            units="galactic",
            alpha=25 * u.deg,
            R=Rs,
        )
        bar_mw["disk"] = mw["disk"].replicate(m=4.1e10 * u.Msun)
        bar_mw["halo"] = mw["halo"]
        bar_mw["nucleus"] = mw["nucleus"]

        # Static bar potential in rotating frame
        bar_mw_static = gp.CCompositePotential()
        bar_mw_static["bar"] = gp.LongMuraliBarPotential(
            m=1e10 * u.Msun,
            a=4 * u.kpc,
            b=0.8 * u.kpc,
            c=0.25 * u.kpc,
            alpha=25 * u.deg,
            units="galactic",
        )
        bar_mw_static["disk"] = mw["disk"].replicate(m=4.1e10 * u.Msun)
        bar_mw_static["halo"] = mw["halo"]
        bar_mw_static["nucleus"] = mw["nucleus"]

        # Rotating frame
        frame = gp.ConstantRotatingFrame(
            Omega=[0, 0, Omega.value] * Omega.unit, units="galactic"
        )
        bar_H_frame = gp.Hamiltonian(potential=bar_mw_static, frame=frame)

        return {
            "bar_mw": bar_mw,
            "bar_H_frame": bar_H_frame,
            "time_knots": time_knots,
            "Omega": Omega,
            "bar_mw_static": bar_mw_static,
        }

    @pytest.fixture
    def corotation_initial_conditions(self, setup_potentials):
        """Find initial conditions near the corotation radius."""
        import scipy.optimize as so

        bar_mw_static = setup_potentials["bar_mw_static"]
        Omega = setup_potentials["Omega"]

        def func(r):
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                Om = bar_mw_static.circular_velocity([r[0], 0, 0] * u.kpc)[0] / (
                    r[0] * u.kpc
                )
                return (Om - Omega).to(Omega.unit).value ** 2

        res = so.minimize(func, x0=10.0, method="powell")

        r_corot = res.x[0] * u.kpc
        v_circ = Omega * r_corot * u.kpc

        return gd.PhaseSpacePosition(
            pos=[r_corot.value, 0, 0] * r_corot.unit,
            vel=[0, v_circ.value, 0.0] * v_circ.unit,
        )

    def test_rotating_frame_vs_inertial_frame(
        self, setup_potentials, corotation_initial_conditions
    ):
        """
        Test that orbits in rotating frame match transformed inertial frame orbits.

        This test integrates an orbit at the corotation radius in two ways:
        1. In a rotating frame with a static bar potential
        2. In an inertial frame with a time-dependent rotating bar potential

        The orbit from (2) is then transformed to the rotating frame and should
        match the orbit from (1) to within numerical precision.
        """
        bar_mw = setup_potentials["bar_mw"]
        bar_H_frame = setup_potentials["bar_H_frame"]
        time_knots = setup_potentials["time_knots"]
        w0 = corotation_initial_conditions

        # Integrate in rotating frame with static bar
        orbit_rot_frame = bar_H_frame.integrate_orbit(
            w0,
            t1=time_knots.min(),
            t2=time_knots.max(),
            dt=0.1 * u.Myr,
            Integrator="dopri853",
            Integrator_kwargs={"atol": 1e-14, "rtol": 1e-14},
        )

        # Integrate in inertial frame with time-dependent bar
        orbit_inertial = bar_mw.integrate_orbit(
            w0,
            t1=time_knots.min(),
            t2=time_knots.max(),
            dt=0.1 * u.Myr,
            Integrator="dopri853",
            Integrator_kwargs={"atol": 1e-14, "rtol": 1e-14},
        )

        # Transform inertial orbit to rotating frame
        orbit_inertial_in_rot_frame = orbit_inertial.to_frame(bar_H_frame.frame)

        assert orbit_rot_frame.shape == orbit_inertial_in_rot_frame.shape
        assert u.allclose(
            orbit_inertial_in_rot_frame.xyz,
            orbit_rot_frame.xyz,
            rtol=5e-5,
            atol=2e-3 * u.kpc,
        )
        assert u.allclose(
            orbit_inertial_in_rot_frame.v_xyz,
            orbit_rot_frame.v_xyz,
            rtol=5e-5,
            atol=3e-5 * u.kpc / u.Myr,
        )

    def test_energy_conservation(self, setup_potentials, corotation_initial_conditions):
        """
        Test that energy is conserved during orbit integration.

        For both the rotating frame and inertial frame integrations,
        the energy should be conserved to within the numerical tolerance
        of the integrator.
        """
        bar_mw = setup_potentials["bar_mw"]
        bar_H_frame = setup_potentials["bar_H_frame"]
        time_knots = setup_potentials["time_knots"]
        w0 = corotation_initial_conditions

        # Integrate in rotating frame
        orbit_rot_frame = bar_H_frame.integrate_orbit(
            w0,
            t1=time_knots.min(),
            t2=time_knots.max(),
            dt=0.1 * u.Myr,
            Integrator="dopri853",
            Integrator_kwargs={"atol": 1e-14, "rtol": 1e-14},
        )

        # Integrate in inertial frame
        orbit_inertial = bar_mw.integrate_orbit(
            w0,
            t1=time_knots.min(),
            t2=time_knots.max(),
            dt=0.1 * u.Myr,
            Integrator="dopri853",
            Integrator_kwargs={"atol": 1e-14, "rtol": 1e-14},
        )

        orbit_inertial_in_rot_frame = orbit_inertial.to_frame(bar_H_frame.frame)

        # compute jacobi energies:
        E_rot = bar_H_frame.energy(orbit_rot_frame)
        E_inertial = bar_H_frame.energy(orbit_inertial_in_rot_frame)

        # check fractional energy conservation
        frac_E_rot = np.abs((E_rot[1:] - E_rot[0]) / E_rot[0])
        frac_E_inertial = np.abs((E_inertial[1:] - E_inertial[0]) / E_inertial[0])

        assert frac_E_rot.max() < 1e-12, (
            f"Rotating frame energy not conserved: max error = {frac_E_rot.max()}"
        )
        assert frac_E_inertial.max() < 1e-6, (
            f"Inertial frame energy not conserved: max error = {frac_E_inertial.max()}"
        )
