"""
Tests for potential symmetry coordinate support.
"""

import astropy.units as u
import numpy as np
import pytest

from gala.potential import HernquistPotential, MiyamotoNagaiPotential, PlummerPotential
from gala.potential.potential.symmetry import (
    CylindricalSymmetry,
    SphericalSymmetry,
)


class TestSphericalSymmetry:
    """Tests for SphericalSymmetry class."""

    def test_to_cartesian_scalar(self):
        """Test conversion of scalar radius to Cartesian."""
        sym = SphericalSymmetry()
        r = 5.0
        xyz = sym.to_cartesian(r)

        assert xyz.shape == (3, 1)
        assert xyz[0, 0] == 5.0
        assert xyz[1, 0] == 0.0
        assert xyz[2, 0] == 0.0

    def test_to_cartesian_array(self):
        """Test conversion of array of radii to Cartesian."""
        sym = SphericalSymmetry()
        r = np.array([1.0, 2.0, 3.0])
        xyz = sym.to_cartesian(r)

        assert xyz.shape == (3, 3)
        np.testing.assert_array_equal(xyz[0], r)
        np.testing.assert_array_equal(xyz[1], 0.0)
        np.testing.assert_array_equal(xyz[2], 0.0)

    def test_to_cartesian_with_units(self):
        """Test conversion preserves units."""
        sym = SphericalSymmetry()
        r = np.array([1.0, 2.0, 3.0]) * u.kpc
        xyz = sym.to_cartesian(r)

        assert xyz.shape == (3, 3)
        assert xyz.unit == u.kpc
        np.testing.assert_array_equal(xyz[0].value, r.value)

    def test_validate_negative_radius(self):
        """Test validation catches negative radii."""
        sym = SphericalSymmetry()
        r = np.array([1.0, -2.0, 3.0])

        with pytest.raises(ValueError, match="non-negative"):
            sym.validate_coords(r=r)


class TestCylindricalSymmetry:
    """Tests for CylindricalSymmetry class."""

    def test_to_cartesian_scalar(self):
        """Test conversion of scalar R, z to Cartesian."""
        sym = CylindricalSymmetry()
        R = 5.0
        z = 1.0
        xyz = sym.to_cartesian(R, z)

        assert xyz.shape == (3, 1)
        assert xyz[0, 0] == 5.0
        assert xyz[1, 0] == 0.0
        assert xyz[2, 0] == 1.0

    def test_to_cartesian_array(self):
        """Test conversion of arrays to Cartesian."""
        sym = CylindricalSymmetry()
        R = np.array([1.0, 2.0, 3.0])
        z = np.array([0.1, 0.2, 0.3])
        xyz = sym.to_cartesian(R, z)

        assert xyz.shape == (3, 3)
        np.testing.assert_array_equal(xyz[0], R)
        np.testing.assert_array_equal(xyz[1], 0.0)
        np.testing.assert_array_equal(xyz[2], z)

    def test_to_cartesian_default_z(self):
        """Test that z defaults to zero."""
        sym = CylindricalSymmetry()
        R = np.array([1.0, 2.0, 3.0])
        xyz = sym.to_cartesian(R)

        assert xyz.shape == (3, 3)
        np.testing.assert_array_equal(xyz[0], R)
        np.testing.assert_array_equal(xyz[1], 0.0)
        np.testing.assert_array_equal(xyz[2], 0.0)

    def test_to_cartesian_scalar_broadcast(self):
        """Test broadcasting of scalar z to array R."""
        sym = CylindricalSymmetry()
        R = np.array([1.0, 2.0, 3.0])
        z = 0.5
        xyz = sym.to_cartesian(R, z)

        assert xyz.shape == (3, 3)
        np.testing.assert_array_equal(xyz[0], R)
        np.testing.assert_array_equal(xyz[2], 0.5)

    def test_to_cartesian_with_units(self):
        """Test conversion preserves units."""
        sym = CylindricalSymmetry()
        R = np.array([1.0, 2.0, 3.0]) * u.kpc
        z = np.array([0.1, 0.2, 0.3]) * u.kpc
        xyz = sym.to_cartesian(R, z)

        assert xyz.shape == (3, 3)
        assert xyz.unit == u.kpc

    def test_to_cartesian_incompatible_shapes(self):
        """Test that incompatible shapes raise an error."""
        sym = CylindricalSymmetry()
        R = np.array([1.0, 2.0, 3.0])
        z = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="Incompatible shapes"):
            sym.to_cartesian(R, z)

    def test_validate_negative_R(self):
        """Test validation catches negative R."""
        sym = CylindricalSymmetry()
        R = np.array([1.0, -2.0, 3.0])

        with pytest.raises(ValueError, match="non-negative"):
            sym.validate_coords(R=R)


class TestSphericalPotentialWithSymmetry:
    """Test spherical potentials using symmetry coordinates."""

    def setup_method(self):
        """Set up test potentials."""
        self.pot_hernquist = HernquistPotential(
            m=1e10 * u.Msun, c=1 * u.kpc, units=[u.kpc, u.Myr, u.Msun, u.radian]
        )
        self.pot_plummer = PlummerPotential(
            m=1e10 * u.Msun, b=1 * u.kpc, units=[u.kpc, u.Myr, u.Msun, u.radian]
        )

    def test_energy_spherical_vs_cartesian(self):
        """Test that energy computed with r matches Cartesian."""
        r = np.array([1.0, 2.0, 5.0, 10.0]) * u.kpc

        # Using spherical coordinate
        E_r = self.pot_hernquist.energy(r=r)

        # Using Cartesian (x, 0, 0)
        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        E_xyz = self.pot_hernquist.energy(xyz)

        np.testing.assert_allclose(E_r.value, E_xyz.value, rtol=1e-10)

    def test_gradient_spherical_vs_cartesian(self):
        """Test that gradient computed with r matches Cartesian."""
        r = np.array([1.0, 2.0, 5.0, 10.0]) * u.kpc

        # Using spherical coordinate
        grad_r = self.pot_hernquist.gradient(r=r)

        # Using Cartesian (x, 0, 0)
        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        grad_xyz = self.pot_hernquist.gradient(xyz)

        np.testing.assert_allclose(grad_r.value, grad_xyz.value, rtol=1e-10)

    def test_density_spherical(self):
        """Test density computation with spherical coordinates."""
        r = np.array([0.5, 1.0, 2.0]) * u.kpc

        # Using spherical coordinate
        rho_r = self.pot_hernquist.density(r=r)

        # Using Cartesian
        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        rho_xyz = self.pot_hernquist.density(xyz)

        np.testing.assert_allclose(rho_r.value, rho_xyz.value, rtol=1e-10)

    def test_acceleration_spherical(self):
        """Test acceleration with spherical coordinates."""
        r = np.array([1.0, 5.0, 10.0]) * u.kpc

        acc_r = self.pot_hernquist.acceleration(r=r)

        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        acc_xyz = self.pot_hernquist.acceleration(xyz)

        np.testing.assert_allclose(acc_r.value, acc_xyz.value, rtol=1e-10)

    def test_mass_enclosed_spherical(self):
        """Test mass_enclosed with spherical coordinates."""
        r = np.array([1.0, 5.0, 10.0]) * u.kpc

        m_r = self.pot_hernquist.mass_enclosed(r=r)

        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        m_xyz = self.pot_hernquist.mass_enclosed(xyz)

        np.testing.assert_allclose(m_r.value, m_xyz.value, rtol=1e-8)

    def test_circular_velocity_spherical(self):
        """Test circular_velocity with spherical coordinates."""
        r = np.array([1.0, 5.0, 10.0]) * u.kpc

        v_r = self.pot_hernquist.circular_velocity(r=r)

        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        v_xyz = self.pot_hernquist.circular_velocity(xyz)

        np.testing.assert_allclose(v_r.value, v_xyz.value, rtol=1e-8)

    def test_hessian_spherical(self):
        """Test hessian with spherical coordinates."""
        r = np.array([1.0, 5.0]) * u.kpc

        H_r = self.pot_hernquist.hessian(r=r)

        xyz = np.zeros((3, len(r))) * u.kpc
        xyz[0] = r
        H_xyz = self.pot_hernquist.hessian(xyz)

        np.testing.assert_allclose(H_r.value, H_xyz.value, rtol=1e-10)

    def test_scalar_input(self):
        """Test that scalar inputs work."""
        r_scalar = 5.0 * u.kpc

        E_scalar = self.pot_hernquist.energy(r=r_scalar)
        # Scalars get converted to shape (1,) internally, which is fine
        assert E_scalar.shape == (1,)
        assert E_scalar.size == 1

    def test_multiple_potentials(self):
        """Test with different spherical potentials."""
        r = np.array([1.0, 5.0, 10.0]) * u.kpc

        E_hern = self.pot_hernquist.energy(r=r)
        E_plum = self.pot_plummer.energy(r=r)

        # Just check they run and produce different results
        assert not np.allclose(E_hern.value, E_plum.value)


class TestCylindricalPotentialWithSymmetry:
    """Test cylindrical potentials using symmetry coordinates."""

    def setup_method(self):
        """Set up test potentials."""
        self.pot = MiyamotoNagaiPotential(
            m=1e11 * u.Msun,
            a=3 * u.kpc,
            b=0.3 * u.kpc,
            units=[u.kpc, u.Myr, u.Msun, u.radian],
        )

    def test_energy_cylindrical_vs_cartesian(self):
        """Test that energy computed with R,z matches Cartesian."""
        R = np.array([1.0, 5.0, 10.0]) * u.kpc
        z = np.array([0.0, 0.5, 1.0]) * u.kpc

        # Using cylindrical coordinates
        E_cyl = self.pot.energy(R=R, z=z)

        # Using Cartesian (R, 0, z)
        xyz = np.zeros((3, len(R))) * u.kpc
        xyz[0] = R
        xyz[2] = z
        E_xyz = self.pot.energy(xyz)

        np.testing.assert_allclose(E_cyl.value, E_xyz.value, rtol=1e-10)

    def test_energy_default_z(self):
        """Test that z defaults to zero."""
        R = np.array([1.0, 5.0, 10.0]) * u.kpc

        # Using R only (z defaults to 0)
        E_R = self.pot.energy(R=R)

        # Using R with explicit z=0
        E_Rz = self.pot.energy(R=R, z=0 * u.kpc)

        np.testing.assert_allclose(E_R.value, E_Rz.value, rtol=1e-10)

    def test_gradient_cylindrical(self):
        """Test gradient with cylindrical coordinates."""
        R = np.array([5.0, 8.0]) * u.kpc
        z = np.array([0.2, 0.5]) * u.kpc

        grad_cyl = self.pot.gradient(R=R, z=z)

        xyz = np.zeros((3, len(R))) * u.kpc
        xyz[0] = R
        xyz[2] = z
        grad_xyz = self.pot.gradient(xyz)

        np.testing.assert_allclose(grad_cyl.value, grad_xyz.value, rtol=1e-10)

    def test_density_cylindrical(self):
        """Test density with cylindrical coordinates."""
        R = np.array([5.0, 8.0]) * u.kpc
        z = np.array([0.2, 0.5]) * u.kpc

        rho_cyl = self.pot.density(R=R, z=z)

        xyz = np.zeros((3, len(R))) * u.kpc
        xyz[0] = R
        xyz[2] = z
        rho_xyz = self.pot.density(xyz)

        np.testing.assert_allclose(rho_cyl.value, rho_xyz.value, rtol=1e-10)

    def test_acceleration_cylindrical(self):
        """Test acceleration with cylindrical coordinates."""
        R = np.array([5.0, 8.0]) * u.kpc
        z = np.array([0.0, 0.5]) * u.kpc

        acc_cyl = self.pot.acceleration(R=R, z=z)

        xyz = np.zeros((3, len(R))) * u.kpc
        xyz[0] = R
        xyz[2] = z
        acc_xyz = self.pot.acceleration(xyz)

        np.testing.assert_allclose(acc_cyl.value, acc_xyz.value, rtol=1e-10)

    def test_scalar_broadcast(self):
        """Test broadcasting of scalar z to array R."""
        R = np.array([5.0, 8.0, 10.0]) * u.kpc
        z = 0.5 * u.kpc

        E = self.pot.energy(R=R, z=z)
        assert E.shape == (3,)

    def test_midplane_values(self):
        """Test midplane (z=0) calculations."""
        R = np.linspace(1, 15, 10) * u.kpc

        # Explicit z=0
        E_z0 = self.pot.energy(R=R, z=0 * u.kpc)

        # Default z (should be 0)
        E_default = self.pot.energy(R=R)

        np.testing.assert_allclose(E_z0.value, E_default.value)


class TestErrorHandling:
    """Test error handling for symmetry coordinates."""

    def setup_method(self):
        """Set up test potentials."""
        self.pot_spherical = HernquistPotential(
            m=1e10 * u.Msun, c=1 * u.kpc, units=[u.kpc, u.Myr, u.Msun, u.radian]
        )
        self.pot_cylindrical = MiyamotoNagaiPotential(
            m=1e11 * u.Msun,
            a=3 * u.kpc,
            b=0.3 * u.kpc,
            units=[u.kpc, u.Myr, u.Msun, u.radian],
        )

    def test_both_cartesian_and_symmetry_coords(self):
        """Test that providing both raises an error."""
        xyz = np.array([[1.0], [0.0], [0.0]]) * u.kpc
        r = 1.0 * u.kpc

        with pytest.raises(ValueError, match="Cannot provide both"):
            self.pot_spherical.energy(xyz, r=r)

    def test_wrong_symmetry_coords_for_potential(self):
        """Test using cylindrical coords on spherical potential."""
        R = np.array([1.0, 2.0]) * u.kpc

        # HernquistPotential is spherical, doesn't accept R (without r)
        with pytest.raises(ValueError, match="symmetry"):
            self.pot_spherical.energy(R=R)

    def test_missing_required_coords(self):
        """Test error when no position is provided."""
        with pytest.raises(ValueError, match="Must provide"):
            self.pot_spherical.energy()

    def test_no_symmetry_potential(self):
        """Test that potentials without symmetry reject symmetry coords."""
        from gala.potential import LogarithmicPotential

        pot = LogarithmicPotential(
            v_c=150 * u.km / u.s,
            r_h=0,
            q1=1,
            q2=0.9,
            q3=0.8,
            phi=0,
            units=[u.kpc, u.Myr, u.Msun, u.radian],
        )

        r = 5.0 * u.kpc

        # LogarithmicPotential is not spherically symmetric (q2, q3 != 1)
        # so it shouldn't accept r= coordinate
        with pytest.raises(ValueError, match="does not have a defined symmetry"):
            pot.energy(r=r)
