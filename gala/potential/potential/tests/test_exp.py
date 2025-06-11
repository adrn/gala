"""
Test the EXP potential
"""

import os

# Third-party
import astropy.units as u
import pytest

# This project
from ...._cconfig import EXP_ENABLED
from ..builtin import EXPPotential
from .helpers import PotentialTestBase
from ....units import SimulationUnitSystem

EXP_CONFIG_FILE = "exp_basis.yml"
EXP_COEFF_FILE = "exp_hernquist_coefs.h5"

# Use in CI to ensure tests aren't silently skipped
FORCE_EXP_TEST = os.environ.get("GALA_FORCE_EXP_TEST", "0") == "1"


@pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)
class TestEXP(PotentialTestBase):
    tol = 1e-3  # increase tolerance for gradient test

    exp_units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)
    potential = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coeff_file=EXP_COEFF_FILE,
        snapshot_index=0,
        units=exp_units,
        stride=1,
    )
    w0 = [
        *[-8, 0.0, 0.0],  #  * u.kpc,
        *[0.0, 180, 0.0]  #  * u.km / u.s,
    ]
    show_plots = True
    check_finite_at_origin = True

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
