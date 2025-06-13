"""
Test the EXP potential
"""

import os

# Third-party
import astropy.units as u
import pytest
from astropy.utils.data import get_pkg_data_filename

import gala.dynamics as gd
from gala._cconfig import EXP_ENABLED
from gala.potential.potential.builtin import EXPPotential
from gala.potential.potential.tests.helpers import PotentialTestBase
from gala.units import SimulationUnitSystem
from gala.util import chdir

EXP_CONFIG_FILE = get_pkg_data_filename("EXP-Hernquist-basis.yml")
EXP_SINGLE_COEF_FILE = get_pkg_data_filename("EXP-Hernquist-single-coefs.hdf5")
EXP_MULTI_COEF_FILE = get_pkg_data_filename("EXP-Hernquist-multi-coefs.hdf5")

# Use in CI to ensure tests aren't silently skipped
FORCE_EXP_TEST = os.environ.get("GALA_FORCE_EXP_TEST", "0") == "1"

# See: generate_exp.py, which generates the basis and coefficients for these tests


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
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

    def setup_method(self):
        assert os.path.exists(self.EXP_CONFIG_FILE), "EXP config file does not exist"
        assert os.path.exists(self.EXP_COEF_FILE), "EXP coef file does not exist"

        with chdir(os.path.dirname(EXP_CONFIG_FILE)):
            self.potential = EXPPotential(
                config_file=self.EXP_CONFIG_FILE,
                coef_file=self.EXP_COEF_FILE,
                snapshot_index=0,
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


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
class TestEXPSingle(EXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_SINGLE_COEF_FILE


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
class TestEXPMulti(EXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_MULTI_COEF_FILE


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
def test_exp_unit_tests():
    with chdir(os.path.dirname(EXP_CONFIG_FILE)):
        pot_single = EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=get_pkg_data_filename("EXP-Hernquist-single-coefs.hdf5"),
            snapshot_index=0,
            units=EXPTestBase.exp_units,
        )

        pot_multi = EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=get_pkg_data_filename("EXP-Hernquist-multi-coefs.hdf5"),
            units=EXPTestBase.exp_units,
        )

        pot_multi_frozen = EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=get_pkg_data_filename("EXP-Hernquist-multi-coefs.hdf5"),
            snapshot_index=0,
            units=EXPTestBase.exp_units,
        )

        test_x = [8.0, 0, 0] * u.kpc
        assert u.allclose(
            pot_single.energy(test_x, t=0 * u.Gyr),
            pot_single.energy(test_x, t=1.4 * u.Gyr),
        )
        assert not u.allclose(
            pot_multi.energy(test_x, t=0 * u.Gyr),
            pot_multi.energy(test_x, t=1.4 * u.Gyr),
        )
        assert u.allclose(
            pot_multi_frozen.energy(test_x, t=0 * u.Gyr),
            pot_multi_frozen.energy(test_x, t=1.4 * u.Gyr),
        )


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
def test_exception():
    """Test that an exception is raised if the config file is not found."""
    with chdir(os.path.dirname(EXP_CONFIG_FILE)):
        with pytest.raises(RuntimeError):
            EXPPotential(
                config_file="nonexistent_config.yml",
                coef_file=EXP_SINGLE_COEF_FILE,
                snapshot_index=0,
                units=SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1),
            )

        with pytest.raises(RuntimeError):
            EXPPotential(
                config_file=EXP_CONFIG_FILE,
                coef_file=EXP_SINGLE_COEF_FILE,
                snapshot_index=0xBAD,
                units=SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1),
            )

        with pytest.raises(RuntimeError):
            EXPPotential(
                config_file=EXP_CONFIG_FILE,
                coef_file=EXP_SINGLE_COEF_FILE,
                tmin=0xBAD,
                units=SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1),
            )
