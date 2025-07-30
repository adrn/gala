"""
Test the builtin CPotential classes
"""

import astropy.table as at
import astropy.units as u
import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_filename
from scipy.spatial.transform import Rotation

from gala._cconfig import GSL_ENABLED
from gala.tests.optional_deps import HAS_SYMPY

from ....units import DimensionlessUnitSystem, galactic, solarsystem
from ...frame import ConstantRotatingFrame
from .. import builtin as p
from ..ccompositepotential import CCompositePotential
from ..core import CompositePotential
from .helpers import CompositePotentialTestBase, PotentialTestBase

##############################################################################
# Python
##############################################################################


class TestHarmonicOscillator1D(PotentialTestBase):
    potential = p.HarmonicOscillatorPotential(omega=1.0)
    w0 = [1.0, 0.1]
    sympy_density = False
    check_finite_at_origin = False
    check_zero_at_infinity = False

    def test_plot(self):
        # Skip for now because contour plotting assumes 3D
        pass


class TestHarmonicOscillator2D(PotentialTestBase):
    potential = p.HarmonicOscillatorPotential(omega=[1.0, 2])
    w0 = [1.0, 0.5, 0.0, 0.1]
    sympy_density = False
    check_finite_at_origin = False
    check_zero_at_infinity = False

    def test_plot(self):
        # Skip for now because contour plotting assumes 3D
        pass

    @pytest.mark.skip(reason="to_sympy() won't support multi-dim HO")
    def test_against_sympy(self):
        pass


##############################################################################
# Cython
##############################################################################


class TestNull(PotentialTestBase):
    potential = p.NullPotential()
    w0 = [1.0, 0.0, 0.0, 0.0, 2 * np.pi, 0.0]

    def test_mass_enclosed(self):
        for arr, shp in zip(self.w0s, self._valu_return_shapes):
            g = self.potential.mass_enclosed(arr[: self.ndim])
            assert g.shape == shp
            assert np.all(g == 0.0)

            g = self.potential.mass_enclosed(arr[: self.ndim], t=0.1)
            g = self.potential.mass_enclosed(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.mass_enclosed(arr[: self.ndim], t=t)
            g = self.potential.mass_enclosed(
                arr[: self.ndim], t=t * self.potential.units["time"]
            )

    def test_circular_velocity(self):
        for arr, shp in zip(self.w0s, self._valu_return_shapes):
            g = self.potential.circular_velocity(arr[: self.ndim])
            assert g.shape == shp
            assert np.all(g == 0.0)

            g = self.potential.circular_velocity(arr[: self.ndim], t=0.1)
            g = self.potential.circular_velocity(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.circular_velocity(arr[: self.ndim], t=t)
            g = self.potential.circular_velocity(
                arr[: self.ndim], t=t * self.potential.units["time"]
            )

    @pytest.mark.skip(reason="Nothing to compare to for Null potential!")
    def test_against_sympy(self):
        pass


class TestHenonHeiles(PotentialTestBase):
    potential = p.HenonHeilesPotential()
    w0 = [1.0, 0.0, 0.0, 2 * np.pi]
    sympy_density = False
    check_finite_at_origin = False
    check_zero_at_infinity = False

    @pytest.mark.skip(reason="Not relevant")
    def test_plot(self):
        pass


class TestKepler(PotentialTestBase):
    potential = p.KeplerPotential(units=solarsystem, m=1.0)
    w0 = [1.0, 0.0, 0.0, 0.0, 2 * np.pi, 0.0]
    # show_plots = True
    check_finite_at_origin = False


class TestKeplerUnitInput(PotentialTestBase):
    potential = p.KeplerPotential(units=solarsystem, m=(1 * u.Msun).to(u.Mjup))
    w0 = [1.0, 0.0, 0.0, 0.0, 2 * np.pi, 0.0]
    check_finite_at_origin = False


class TestIsochrone(PotentialTestBase):
    potential = p.IsochronePotential(units=solarsystem, m=1.0, b=0.1)
    w0 = [1.0, 0.0, 0.0, 0.0, 2 * np.pi, 0.0]


class TestIsochroneDimensionless(PotentialTestBase):
    potential = p.IsochronePotential(units=DimensionlessUnitSystem(), m=1.0, b=0.1)
    w0 = [1.0, 0.0, 0.0, 0.0, 2 * np.pi, 0.0]


class TestHernquist(PotentialTestBase):
    potential = p.HernquistPotential(units=galactic, m=1.0e11, c=0.26)
    w0 = [1.0, 0.0, 0.0, 0.0, 0.1, 0.1]


class TestPlummer(PotentialTestBase):
    potential = p.PlummerPotential(units=galactic, m=1.0e11, b=0.26)
    w0 = [1.0, 0.0, 0.0, 0.0, 0.1, 0.1]


class TestJaffe(PotentialTestBase):
    check_finite_at_origin = False
    potential = p.JaffePotential(units=galactic, m=1.0e11, c=0.26)
    w0 = [1.0, 0.0, 0.0, 0.0, 0.1, 0.1]


class TestMiyamotoNagai(PotentialTestBase):
    potential = p.MiyamotoNagaiPotential(units=galactic, m=1.0e11, a=6.5, b=0.26)
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    rotation = True

    @pytest.mark.skipif(not HAS_SYMPY, reason="requires sympy to run this test")
    def test_hessian_analytic(self):
        import sympy as sy
        from astropy.constants import G
        from sympy import symbols

        x, y, z = symbols("x y z")

        usys = self.potential.units
        GM = (G * self.potential.parameters["m"]).decompose(usys).value
        a = self.potential.parameters["a"].decompose(usys).value
        b = self.potential.parameters["b"].decompose(usys).value
        Phi = -GM / sy.sqrt(x**2 + y**2 + (a + sy.sqrt(z**2 + b**2)) ** 2)

        d2Phi_dx2 = sy.lambdify((x, y, z), sy.diff(Phi, x, 2))
        d2Phi_dy2 = sy.lambdify((x, y, z), sy.diff(Phi, y, 2))
        d2Phi_dz2 = sy.lambdify((x, y, z), sy.diff(Phi, z, 2))

        d2Phi_dxdy = sy.lambdify((x, y, z), sy.diff(Phi, x, y))
        d2Phi_dxdz = sy.lambdify((x, y, z), sy.diff(Phi, x, z))
        d2Phi_dydz = sy.lambdify((x, y, z), sy.diff(Phi, y, z))

        rnd = np.random.default_rng(42)
        xyz = rnd.normal(0, 25, size=(3, 64))

        H1 = self.potential.hessian(xyz).decompose(usys).value

        H2 = np.zeros((3, 3, xyz.shape[1]))
        H2[0, 0] = d2Phi_dx2(*xyz)
        H2[1, 1] = d2Phi_dy2(*xyz)
        H2[2, 2] = d2Phi_dz2(*xyz)

        H2[0, 1] = H2[1, 0] = d2Phi_dxdy(*xyz)
        H2[0, 2] = H2[2, 0] = d2Phi_dxdz(*xyz)
        H2[1, 2] = H2[2, 1] = d2Phi_dydz(*xyz)

        assert np.allclose(H1, H2)


class TestMN3(PotentialTestBase):
    potential = p.MN3ExponentialDiskPotential(
        units=galactic, m=1.0e11, h_R=3.5, h_z=0.26
    )
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    rotation = True

    # TODO:
    @pytest.mark.skip(reason="to_sympy() not implemented yet")
    def test_against_sympy(self):
        pass

    def test_get_three(self):
        pots = self.potential.get_three_potentials()
        assert len(pots) == 3


class TestSatoh(PotentialTestBase):
    potential = p.SatohPotential(units=galactic, m=1.0e11, a=6.5, b=0.26)
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    rotation = True


class TestKuzmin(PotentialTestBase):
    potential = p.KuzminPotential(units=galactic, m=1.0e11, a=3.5)
    w0 = [8.0, 0.0, 0.0, 0.0, 0.22, 0.1]
    sympy_hessian = False
    sympy_density = False
    rotation = True


class TestStone(PotentialTestBase):
    potential = p.StonePotential(units=galactic, m=1e11, r_c=0.1, r_h=10.0)
    w0 = [8.0, 0.0, 0.0, 0.0, 0.18, 0.1]


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
class TestPowerLawCutoff(PotentialTestBase):
    w0 = [8.0, 0.0, 0.0, 0.0, 0.1, 0.1]
    atol = 1e-3
    sympy_density = False  # weird underflow issues??
    check_finite_at_origin = False

    def setup_method(self):
        self.potential = p.PowerLawCutoffPotential(
            units=galactic, m=1e10, r_c=1.0, alpha=1.8
        )
        super().setup_method()


class TestSphericalNFW(PotentialTestBase):
    potential = p.NFWPotential(units=galactic, m=1e11, r_s=12.0)
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]


class TestFlattenedNFW(PotentialTestBase):
    potential = p.NFWPotential(units=galactic, m=1e11, r_s=12.0, c=0.7)
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]
    sympy_density = False  # not defined
    rotation = True

    def test_against_spherical(self):
        """
        Note: This is a regression test for Issue #254
        """

        sph = p.NFWPotential(units=galactic, m=1e11, r_s=12.0)
        assert not u.allclose(
            self.potential.gradient(self.w0[:3]), sph.gradient(self.w0[:3])
        )


class TestTriaxialNFW(PotentialTestBase):
    potential = p.NFWPotential(units=galactic, m=1e11, r_s=12.0, a=1.0, b=0.95, c=0.9)
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]
    sympy_density = False  # not defined
    rotation = True


class TestSphericalNFWFromCircVel(PotentialTestBase):
    potential = p.NFWPotential.from_circular_velocity(
        v_c=220.0 * u.km / u.s, r_s=20 * u.kpc, r_ref=8.0 * u.kpc, units=galactic
    )
    w0 = [19.0, 2.7, -0.9, 0.00352238, -0.165134, 0.0075]

    def test_circ_vel(self):
        for r_ref in [3.0, 8.0, 21.7234]:
            pot = p.NFWPotential.from_circular_velocity(
                v_c=220.0 * u.km / u.s,
                r_s=20 * u.kpc,
                r_ref=r_ref * u.kpc,
                units=galactic,
            )
            vc = pot.circular_velocity([r_ref, 0, 0] * u.kpc)  # at ref. velocity
            assert u.allclose(vc, 220 * u.km / u.s)

    def test_against_triaxial(self):
        this = p.NFWPotential.from_circular_velocity(
            v_c=220.0 * u.km / u.s, r_s=20 * u.kpc, units=galactic
        )
        other = p.LeeSutoTriaxialNFWPotential(
            units=galactic,
            v_c=220.0 * u.km / u.s,
            r_s=20.0 * u.kpc,
            a=1.0,
            b=1.0,
            c=1.0,
        )

        v1 = this.energy(self.w0[:3])
        v2 = other.energy(self.w0[:3])
        assert u.allclose(v1, v2)

        a1 = this.gradient(self.w0[:3])
        a2 = other.gradient(self.w0[:3])
        assert u.allclose(a1, a2)

        d1 = this.density(self.w0[:3])
        d2 = other.density(self.w0[:3])
        assert u.allclose(d1, d2)

    def test_mass_enclosed(self):
        # true mass profile
        m = self.potential.parameters["m"].value
        rs = self.potential.parameters["r_s"].value

        r = np.linspace(1.0, 400, 100)
        fac = np.log(1 + r / rs) - (r / rs) / (1 + (r / rs))
        true_mprof = m * fac

        R = np.zeros((3, len(r)))
        R[0, :] = r
        esti_mprof = self.potential.mass_enclosed(R)

        assert np.allclose(true_mprof, esti_mprof.value, rtol=1e-6)


class TestNFW(PotentialTestBase):
    potential = p.NFWPotential(
        m=6e11 * u.Msun, r_s=20 * u.kpc, a=1.0, b=0.9, c=0.75, units=galactic
    )
    w0 = [19.0, 2.7, -0.9, 0.00352238, -0.15134, 0.0075]
    sympy_density = False  # like triaxial case

    def test_compare(self):
        sph = p.NFWPotential(m=6e11 * u.Msun, r_s=20 * u.kpc, units=galactic)
        fla = p.NFWPotential(m=6e11 * u.Msun, r_s=20 * u.kpc, c=0.8, units=galactic)
        tri = p.NFWPotential(
            m=6e11 * u.Msun, r_s=20 * u.kpc, b=0.9, c=0.8, units=galactic
        )

        xyz = np.zeros((3, 128))
        xyz[0] = np.logspace(-1.0, 3, xyz.shape[1])

        assert u.allclose(sph.energy(xyz), fla.energy(xyz))
        assert u.allclose(sph.energy(xyz), tri.energy(xyz))

        assert u.allclose(sph.gradient(xyz), fla.gradient(xyz))
        assert u.allclose(sph.gradient(xyz), tri.gradient(xyz))

        # assert u.allclose(sph.density(xyz), fla.density(xyz)) # TODO: fla density not implemented
        # assert u.allclose(sph.density(xyz), tri.density(xyz)) # TODO: tri density not implemented

        # ---

        tri = p.NFWPotential(
            m=6e11 * u.Msun, r_s=20 * u.kpc, a=0.9, c=0.8, units=galactic
        )

        xyz = np.zeros((3, 128))
        xyz[1] = np.logspace(-1.0, 3, xyz.shape[1])

        assert u.allclose(sph.energy(xyz), fla.energy(xyz))
        assert u.allclose(sph.energy(xyz), tri.energy(xyz))

        assert u.allclose(sph.gradient(xyz), fla.gradient(xyz))
        assert u.allclose(sph.gradient(xyz), tri.gradient(xyz))

        # assert u.allclose(sph.density(xyz), fla.density(xyz)) # TODO: fla density not implemented
        # assert u.allclose(sph.density(xyz), tri.density(xyz)) # TODO: tri density not implemented

        # ---

        xyz = np.zeros((3, 128))
        xyz[0] = np.logspace(-1.0, 3, xyz.shape[1])
        xyz[1] = np.logspace(-1.0, 3, xyz.shape[1])

        assert u.allclose(sph.energy(xyz), fla.energy(xyz))
        assert u.allclose(sph.gradient(xyz), fla.gradient(xyz))

    def test_nfw_properties(self):
        """Test that M200, c200, and R200 properties work correctly."""

        M200_input = 1e12 * u.Msun
        c_input = 15.0
        pot_input = p.NFWPotential.from_M200_c(M200_input, c_input, units=galactic)

        # Test the inverse properties
        c200_computed = pot_input.c200()
        M200_computed = pot_input.M200()

        assert u.allclose(c200_computed, c_input)
        assert u.allclose(M200_computed, M200_input)

        # Test that R200 evaluates and is equivalent to r_s * c200
        R200_computed = pot_input.R200()
        r_s = pot_input.parameters["r_s"]
        R200_expected = r_s * c200_computed
        assert u.allclose(R200_computed, R200_expected)


class TestLeeSutoTriaxialNFW(PotentialTestBase):
    potential = p.LeeSutoTriaxialNFWPotential(
        units=galactic, v_c=0.35, r_s=12.0, a=1.3, b=1.0, c=0.8
    )
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]
    rotation = True

    @pytest.mark.skip(reason="to_sympy() not implemented yet")
    def test_against_sympy(self):
        pass


class TestLogarithmic(PotentialTestBase):
    potential = p.LogarithmicPotential(
        units=galactic, v_c=0.17, r_h=10.0, q1=1.2, q2=1.0, q3=0.8
    )
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]
    check_zero_at_infinity = False


class TestLongMuraliBar(PotentialTestBase):
    potential = p.LongMuraliBarPotential(
        units=galactic, m=1e11, a=4.0 * u.kpc, b=1 * u.kpc, c=1.0 * u.kpc
    )
    vc = potential.circular_velocity([19.0, 0, 0] * u.kpc).decompose(galactic).value[0]
    w0 = [19.0, 0.2, -0.9, 0.0, vc, 0.0]
    rotation = True


class TestLongMuraliBarRotate(PotentialTestBase):
    potential = p.LongMuraliBarPotential(
        units=galactic,
        m=1e11,
        a=4.0 * u.kpc,
        b=1 * u.kpc,
        c=1.0 * u.kpc,
        R=np.array(
            [
                [0.63302222, 0.75440651, 0.17364818],
                [-0.76604444, 0.64278761, 0.0],
                [-0.1116189, -0.13302222, 0.98480775],
            ]
        ),
    )
    vc = potential.circular_velocity([19.0, 0, 0] * u.kpc).decompose(galactic).value[0]
    w0 = [19.0, 0.2, -0.9, 0.0, vc, 0.0]

    def test_hessian(self):
        # TODO: when hessian for rotated potentials implemented, remove this
        with pytest.raises(NotImplementedError):
            self.potential.hessian([1.0, 2.0, 3.0])

    @pytest.mark.skip(reason="Not implemented for rotated potentials")
    def test_against_sympy(self):
        pass


class TestLongMuraliBarRotationScipy(PotentialTestBase):
    potential = p.LongMuraliBarPotential(
        units=galactic,
        m=1e11,
        a=4.0 * u.kpc,
        b=1 * u.kpc,
        c=1.0 * u.kpc,
        R=Rotation.from_euler("zxz", [90.0, 0, 0.0], degrees=True),
    )
    vc = potential.circular_velocity([19.0, 0, 0] * u.kpc).decompose(galactic).value[0]
    w0 = [19.0, 0.2, -0.9, 0.0, vc, 0.0]

    def test_hessian(self):
        # TODO: when hessian for rotated potentials implemented, remove this
        with pytest.raises(NotImplementedError):
            self.potential.hessian([1.0, 2.0, 3.0])

    @pytest.mark.skip(reason="Not implemented for rotated potentials")
    def test_against_sympy(self):
        pass


class TestComposite(CompositePotentialTestBase):
    p1 = p.LogarithmicPotential(
        units=galactic, v_c=0.17, r_h=10.0, q1=1.2, q2=1.0, q3=0.8
    )
    p2 = p.MiyamotoNagaiPotential(units=galactic, m=1.0e11, a=6.5, b=0.26)
    potential = CompositePotential()
    potential["disk"] = p2
    potential["halo"] = p1
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]
    rotation = True
    check_zero_at_infinity = False


class TestCComposite(CompositePotentialTestBase):
    p1 = p.LogarithmicPotential(
        units=galactic, v_c=0.17, r_h=10.0, q1=1.2, q2=1.0, q3=0.8
    )
    p2 = p.MiyamotoNagaiPotential(units=galactic, m=1.0e11, a=6.5, b=0.26)
    potential = CCompositePotential()
    potential["disk"] = p2
    potential["halo"] = p1
    w0 = [19.0, 2.7, -6.9, 0.0352238, -0.03579493, 0.075]
    rotation = True
    check_zero_at_infinity = False


class TestKepler3Body(CompositePotentialTestBase):
    """This implicitly tests the origin shift"""

    mu = 1 / 11.0
    x1 = -mu
    m1 = 1 - mu
    x2 = 1 - mu
    m2 = mu
    potential = CCompositePotential()
    potential["m1"] = p.KeplerPotential(m=m1, origin=[x1, 0, 0.0])
    potential["m2"] = p.KeplerPotential(m=m2, origin=[x2, 0, 0.0])

    Omega = np.array([0, 0, 1.0])
    frame = ConstantRotatingFrame(Omega=Omega)
    w0 = [0.5, 0, 0, 0.0, 1.05800316, 0.0]


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
class TestMultipoleInner(CompositePotentialTestBase):
    potential_1 = p.NFWPotential(m=1e12, r_s=15.0, units=galactic)
    potential = potential_1 + p.MultipolePotential(
        units=galactic, m=1e10, r_s=15.0, inner=True, lmax=2, S10=1.0, S21=0.5
    )
    vc = potential.circular_velocity([19.0, 0, 0] * u.kpc).decompose(galactic).value[0]
    w0 = [19.0, 0.2, -0.9, 0.0, vc, 0.0]
    check_zero_at_infinity = False

    @pytest.mark.skip(reason="Not implemented for multipole potentials")
    def test_hessian(self):
        pass

    @pytest.mark.skip(reason="Not implemented for multipole potentials")
    def test_against_sympy(self):
        pass


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
class TestMultipoleOuter(CompositePotentialTestBase):
    potential_1 = p.NFWPotential(m=1e12, r_s=15.0, units=galactic)
    potential = potential_1 + p.MultipolePotential(
        units=galactic, m=1e10, r_s=15.0, inner=False, lmax=2, S10=1.0, S21=0.5
    )
    vc = potential.circular_velocity([19.0, 0, 0] * u.kpc).decompose(galactic).value[0]
    w0 = [19.0, 0.2, -0.9, 0.0, vc, 0.0]
    check_finite_at_origin = False

    @pytest.mark.skip(reason="Not implemented for multipole potentials")
    def test_hessian(self):
        pass

    @pytest.mark.skip(reason="Not implemented for multipole potentials")
    def test_against_sympy(self):
        pass


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
class TestCylspline(PotentialTestBase):
    check_finite_at_origin = True

    def setup_method(self):
        self.potential = p.CylSplinePotential.from_file(
            get_pkg_data_filename("pot_disk_506151.pot"), units=galactic
        )
        vc = self.potential.circular_velocity([19.0, 0, 0] * u.kpc).decompose(galactic)
        self.w0 = [19.0, 0.2, -0.9, 0.0, vc.value[0], 0.0]
        super().setup_method()

    @pytest.mark.skip(reason="Not implemented for cylspline potentials")
    def test_density(self):
        pass

    @pytest.mark.skip(reason="Not implemented for cylspline potentials")
    def test_hessian(self):
        pass

    @pytest.mark.skip(reason="Not implemented for cylspline potentials")
    def test_against_sympy(self):
        pass

    def test_against_agama(self):
        agama_tbl = at.QTable.read(get_pkg_data_filename("agama_cylspline_test.fits"))

        gala_ene = self.potential.energy(agama_tbl["xyz"].T)
        gala_acc = self.potential.acceleration(agama_tbl["xyz"].T)

        assert u.allclose(gala_ene, agama_tbl["pot"][:, 0], rtol=1e-3)
        for i in range(3):
            assert u.allclose(gala_acc[i], agama_tbl["acc"][:, i], rtol=1e-2)


class TestBurkert(PotentialTestBase):
    potential = p.BurkertPotential(
        units=galactic, rho=5e-25 * u.g / u.cm**3, r0=12 * u.kpc
    )
    w0 = [1.0, 0.0, 0.0, 0.0, 0.1, 0.1]

    check_finite_at_origin = False

    @pytest.mark.skip(reason="Not implemented for Burkert potentials")
    def test_against_sympy(self):
        pass

    @pytest.mark.skip(reason="Hessian not implemented for Burkert potential")
    def test_hessian(self):
        pass

    def test_from_r0(self):
        # Test against values from Zhu+2023
        pot = p.BurkertPotential.from_r0(r0=11.87 * u.kpc, units=galactic)

        rho = pot.parameters["rho"].to(u.g / u.cm**3)
        rho_check = 5.93e-25 * u.g / u.cm**3

        # Check a 1% tolerance on inferred density against published values
        assert abs(rho - rho_check) / rho_check < 0.01
