"""
Test the EXP potential
"""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_filename

import gala.dynamics as gd
import gala.potential as gp
from gala._cconfig import EXP_ENABLED
from gala.potential.potential.builtin import EXPPotential
from gala.potential.potential.tests.helpers import PotentialTestBase
from gala.units import SimulationUnitSystem
from gala.util import chdir

try:
    import pyEXP

    HAVE_PYEXP = True
except ImportError:
    HAVE_PYEXP = False

EXP_CONFIG_FILE = get_pkg_data_filename("EXP-Hernquist-basis.yml")
EXP_SINGLE_COEF_FILE = get_pkg_data_filename("EXP-Hernquist-single-coefs.hdf5")
EXP_MULTI_COEF_FILE = get_pkg_data_filename("EXP-Hernquist-multi-coefs.hdf5")

# Use in CI to ensure tests aren't silently skipped
FORCE_EXP_TEST = os.environ.get("GALA_FORCE_EXP_TEST", "0") == "1"
FORCE_PYEXP_TEST = os.environ.get("GALA_FORCE_PYEXP_TEST", "0") == "1"

# global pytest marker to skip tests if EXP is not enabled
pytestmark = pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)

# See: generate_exp.py, which generates the basis and coefficients for these tests


class EXPTestBase(PotentialTestBase):
    tol = 1e-2  # increase tolerance for gradient test

    exp_units = SimulationUnitSystem(
        mass=1.25234e11 * u.Msun, length=3.845 * u.kpc, G=1
    )
    _tmp = gd.PhaseSpacePosition(
        pos=[-8, 0.0, 0.0] * u.kpc,
        vel=[0.0, 180, 0.0] * u.km / u.s,
    )
    w0 = _tmp.w(exp_units)[:, 0]
    show_plots = False
    check_finite_at_origin = True
    check_zero_at_infinity = False

    def setup_method(self):
        assert os.path.exists(self.EXP_CONFIG_FILE), "EXP config file does not exist"
        assert os.path.exists(self.EXP_COEF_FILE), "EXP coef file does not exist"

        self.potential = EXPPotential(
            config_file=self.EXP_CONFIG_FILE,
            coef_file=self.EXP_COEF_FILE,
            # TODO: this is making the multi-coef test actually static!
            # Need to fix the orbit integration then remove this
            # snapshot_index=0,
            units=self.exp_units,
        )
        return super().setup_method()

    # TODO: deepcopy is not implemented for EXPPotential
    @pytest.mark.skip(reason="Not implemented for EXP")
    def test_unitsystem(self):
        pass

    @pytest.mark.skip(reason="Not implemented for EXP")
    def test_hessian(self):
        pass

    @pytest.mark.skip(reason="Not implemented for EXP")
    def test_against_sympy(self):
        pass

    # TODO: constructing EXPPotential(**other.parameters) is not implemented
    @pytest.mark.skip(reason="Not implemented for EXP")
    def test_compare(self):
        pass

    @pytest.mark.skip(reason="Not implemented for EXP")
    def test_save_load(self):
        pass

    @pytest.mark.skip(reason="Not implemented for EXP")
    def test_pickle(self, tmpdir):
        pass

    def test_orbit_integration(self, *args, **kwargs):
        """Test orbit integration with EXPPotential"""
        if self.potential.static:
            # Use any time range with a static potential.
            time_spec = {}
        else:
            # With a non-static potential, we need to stay within the time range
            time_spec = {"t1": self.potential.tmin_exp, "t2": self.potential.tmax_exp}
        return super().test_orbit_integration(
            *args,
            **kwargs,
            **time_spec,
        )

    @pytest.mark.skipif(
        not FORCE_PYEXP_TEST,
        reason="requires pyEXP",
    )
    def test_pyexp(self):
        """Test EXPPotential against pyEXP"""

        gala_test_x = [1.0, 2.0, -3.0] * u.kpc
        exp_test_x = gala_test_x.to_value(self.exp_units["length"])

        with open(self.EXP_CONFIG_FILE, encoding="utf-8") as fp:
            config_str = fp.read()
        with chdir(os.path.dirname(self.EXP_CONFIG_FILE)):
            exp_basis = pyEXP.basis.Basis.factory(config_str)
        exp_coefs = pyEXP.coefs.Coefs.factory(self.EXP_COEF_FILE)

        # Use a snapshot time so that we don't have to rebuild the interpolation
        # functionality
        t = exp_coefs.Times()[-1] * self.exp_units["time"]

        exp_coefs_at_time = exp_coefs.getCoefStruct(t.to_value(self.exp_units["time"]))
        exp_basis.set_coefs(exp_coefs_at_time)

        exp_fields = exp_basis.getFields(*exp_test_x)
        exp_dens = exp_fields[2] * self.exp_units["mass density"]
        exp_pot = exp_fields[5] * self.exp_units["energy"] / self.exp_units["mass"]
        exp_grad = -np.stack(exp_fields[6:9]) * self.exp_units["acceleration"]

        gala_dens = self.potential.density(gala_test_x, t=t)
        gala_pot = self.potential.energy(gala_test_x, t=t)
        gala_grad = self.potential.gradient(gala_test_x, t=t).reshape(-1)

        assert u.allclose(exp_dens, gala_dens)
        assert u.allclose(exp_pot, gala_pot)
        assert u.allclose(exp_grad, gala_grad)


class TestEXPSingle(EXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_SINGLE_COEF_FILE


class TestEXPMulti(EXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_MULTI_COEF_FILE


def test_exp_unit_tests():
    pot_single = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_SINGLE_COEF_FILE,
        units=EXPTestBase.exp_units,
    )

    pot_single_frozen = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_SINGLE_COEF_FILE,
        snapshot_index=0,
        units=EXPTestBase.exp_units,
    )

    pot_multi = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_FILE,
        units=EXPTestBase.exp_units,
    )

    pot_multi_frozen = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_FILE,
        snapshot_index=0,
        units=EXPTestBase.exp_units,
    )

    # TODO: not yet implemented
    # pot_multi_frozen_arbitrary = EXPPotential(
    #     config_file=EXP_CONFIG_FILE,
    #     coef_file=EXP_MULTI_COEF_FILE,
    #     tmin=0.4 * u.Gyr,
    #     tmax=0.4 * u.Gyr,
    #     units=EXPTestBase.exp_units,
    # )

    assert pot_single.static is True
    assert pot_single_frozen.static is True
    assert pot_multi_frozen.static is True
    # assert pot_multi_frozen_arbitrary.static is True

    assert pot_multi.static is False

    test_x = [8.0, 0, 0] * u.kpc
    assert u.allclose(
        pot_single.energy(test_x, t=0 * u.Gyr),
        pot_single.energy(test_x, t=1.4 * u.Gyr),
    )
    assert u.allclose(
        pot_single_frozen.energy(test_x, t=0 * u.Gyr),
        pot_single_frozen.energy(test_x, t=1.4 * u.Gyr),
    )
    assert not u.allclose(
        pot_multi.energy(test_x, t=0 * u.Gyr),
        pot_multi.energy(test_x, t=1.4 * u.Gyr),
    )
    assert u.allclose(
        pot_multi_frozen.energy(test_x, t=0 * u.Gyr),
        pot_multi_frozen.energy(test_x, t=1.4 * u.Gyr),
    )
    # assert u.allclose(
    #     pot_multi_frozen_arbitrary.energy(test_x, t=0. * u.Gyr),
    #     pot_multi_frozen_arbitrary.energy(test_x, t=1.4 * u.Gyr),
    # )

    # check tmin/tmax
    assert u.allclose(pot_multi.tmin_exp, 0.0 * u.Gyr)
    assert u.allclose(pot_multi.tmax_exp, 2.0 * u.Gyr)


def test_cython_exceptions():
    """Test various exceptions propagated from C++"""
    units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)
    with pytest.raises(RuntimeError):
        EXPPotential(
            config_file="nonexistent_config.yml",
            coef_file=EXP_SINGLE_COEF_FILE,
            snapshot_index=0,
            units=units,
        )

    with pytest.raises(RuntimeError):
        EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=EXP_SINGLE_COEF_FILE,
            snapshot_index=0xBAD,
            units=units,
        )

    with pytest.raises(RuntimeError):
        EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=EXP_MULTI_COEF_FILE,
            tmin=0xBAD,
            units=units,
        )

    pot = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_FILE,
        units=units,
    )

    # TODO: this will eventually be an exception
    assert np.isnan(pot.energy([0, 0, 0], t=float(0xBAD)))


def test_composite():
    """Test that EXPPotential can be used in a CompositePotential"""
    units = SimulationUnitSystem(mass=1.25234e11 * u.Msun, length=3.845 * u.kpc, G=1)
    pot_single = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_SINGLE_COEF_FILE,
        units=units,
    )
    pot_multi = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_FILE,
        units=units,
    )
    composite_pot = pot_single + pot_multi
    assert isinstance(
        composite_pot, gp.potential.ccompositepotential.CCompositePotential
    )

    test_x = [8.0, 0, 0] * u.kpc
    assert u.allclose(
        composite_pot.energy(test_x, t=0 * u.Gyr),
        pot_single.energy(test_x, t=0 * u.Gyr) + pot_multi.energy(test_x, t=0 * u.Gyr),
    )
    assert u.allclose(
        composite_pot.energy(test_x, t=1.4 * u.Gyr),
        pot_single.energy(test_x, t=1.4 * u.Gyr)
        + pot_multi.energy(test_x, t=1.4 * u.Gyr),
    )

    # Test orbit integration
    w0 = gd.PhaseSpacePosition(
        pos=[-8, 0.0, 0.0] * u.kpc,
        vel=[0.0, 220, 0.0] * u.km / u.s,
    )
    orbit = gp.Hamiltonian(composite_pot).integrate_orbit(
        w0, dt=1 * u.Myr, t1=0 * u.Gyr, t2=1 * u.Gyr
    )
    assert orbit is not None
    assert np.all(np.isfinite(orbit.pos.xyz.value))
    assert np.all(np.isfinite(orbit.vel.d_xyz.value))
    assert np.all(np.isfinite(orbit.t.value))


def test_paths():
    """
    Test relative and absolute file paths
    """

    gp.EXPPotential(
        config_file=Path(EXP_CONFIG_FILE).absolute(),
        coef_file=Path(EXP_SINGLE_COEF_FILE).absolute(),
        units=SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1),
    )

    with chdir(Path(EXP_CONFIG_FILE).parent):
        gp.EXPPotential(
            config_file=Path(EXP_CONFIG_FILE).name,
            coef_file=Path(EXP_SINGLE_COEF_FILE).name,
            units=SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1),
        )
