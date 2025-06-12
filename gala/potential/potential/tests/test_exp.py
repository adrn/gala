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

EXP_CONFIG_FILE = get_pkg_data_filename("exp_basis.yml")
EXP_COEFF_FILE = get_pkg_data_filename("exp_hernquist_coefs.h5")

# Use in CI to ensure tests aren't silently skipped
FORCE_EXP_TEST = os.environ.get("GALA_FORCE_EXP_TEST", "0") == "1"


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
class TestEXP(PotentialTestBase):
    tol = 5e-3  # increase tolerance for gradient test

    exp_units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)
    _tmp = gd.PhaseSpacePosition(
        pos=[-8, 0.0, 0.0] * u.kpc,
        vel=[0.0, 180, 0.0] * u.km / u.s,
    )
    w0 = _tmp.w(exp_units)[:, 0]
    show_plots = False
    check_finite_at_origin = True

    def setup_method(self):
        assert os.path.exists(EXP_CONFIG_FILE), "EXP config file does not exist"
        assert os.path.exists(EXP_COEFF_FILE), "EXP coeff file does not exist"

        current_path = os.getcwd()
        os.chdir(os.path.dirname(EXP_CONFIG_FILE))
        self.potential = EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coeff_file=EXP_COEFF_FILE,
            snapshot_index=0,
            units=self.exp_units,
        )
        os.chdir(current_path)
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
