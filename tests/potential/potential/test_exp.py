"""
Test the EXP potential
"""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import EXP_ENABLED
from potential_helpers import PotentialTestBase

import gala.dynamics as gd
import gala.potential as gp
from gala.potential.potential.builtin import EXPPotential, PyEXPPotential
from gala.units import SimulationUnitSystem
from gala.util import chdir

this_path = Path(__file__).parent

# Use in CI to ensure tests aren't silently skipped
FORCE_EXP_TEST = os.environ.get("GALA_FORCE_EXP_TEST", "0") == "1"
FORCE_PYEXP_TEST = os.environ.get("GALA_FORCE_PYEXP_TEST", "0") == "1"

try:
    import pyEXP

    HAVE_PYEXP = True
except ImportError as e:
    HAVE_PYEXP = False
    if FORCE_PYEXP_TEST:
        raise ImportError("pyEXP is required to run pyEXP tests") from e


EXP_CONFIG_FILE = this_path / "EXP-Hernquist-basis.yml"
EXP_FIELD_CONFIG_FILE = this_path / "EXP-field-basis.yml"  # dummy
EXP_SINGLE_COEF_FILE = this_path / "EXP-Hernquist-single-coefs.hdf5"
EXP_MULTI_COEF_FILE = this_path / "EXP-Hernquist-multi-coefs.hdf5"
EXP_MULTI_COEF_SNAPSHOT_TIME_FILE = (
    this_path / "EXP-Hernquist-multi-coefs-snap-time-Gyr.hdf5"
)
EXP_UNITS = SimulationUnitSystem(mass=1.25234e11 * u.Msun, length=3.845 * u.kpc, G=1)

# global pytest marker to skip tests if EXP is not enabled
pytestmark = pytest.mark.skipif(
    not EXP_ENABLED and not FORCE_EXP_TEST,
    reason="requires Gala compiled with EXP support",
)

# See: generate_exp.py, which generates the basis and coefficients for these tests


# base for EXP and PyEXP tests
class CommonEXPTestBase(PotentialTestBase):
    tol = 1e-1  # increase tolerance for gradient test

    exp_units = EXP_UNITS

    _tmp = gd.PhaseSpacePosition(
        pos=[-8, 0.0, 0.0] * u.kpc,
        vel=[0.0, 180, 0.0] * u.km / u.s,
    )
    w0 = _tmp.w(exp_units)[:, 0]
    show_plots = False
    check_finite_at_origin = True
    check_zero_at_infinity = False

    num_dx = 1e-3
    skip_hessian = True

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
        not HAVE_PYEXP,
        reason="requires pyEXP",
    )
    def test_pyexp(self):
        """Test against pyEXP"""

        gala_test_x = [1.0, 2.0, -3.0] * u.kpc
        exp_test_x = gala_test_x.to_value(self.exp_units["length"])

        with open(self.EXP_CONFIG_FILE, encoding="utf-8") as fp:
            config_str = fp.read()
        with chdir(os.path.dirname(self.EXP_CONFIG_FILE)):
            exp_basis = pyEXP.basis.Basis.factory(config_str)
        exp_coefs = pyEXP.coefs.Coefs.factory(str(self.EXP_COEF_FILE))

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


class EXPTestBase(CommonEXPTestBase):
    def setup_method(self):
        assert os.path.exists(self.EXP_CONFIG_FILE), "EXP config file does not exist"
        assert os.path.exists(self.EXP_COEF_FILE), "EXP coef file does not exist"

        self.potential = EXPPotential(
            config_file=self.EXP_CONFIG_FILE,
            coef_file=self.EXP_COEF_FILE,
            units=self.exp_units,
        )
        return super().setup_method()


@pytest.mark.skipif(
    not HAVE_PYEXP,
    reason="requires pyEXP",
)
class PyEXPTestBase(CommonEXPTestBase):
    def setup_method(self):
        assert os.path.exists(self.EXP_CONFIG_FILE), "EXP config file does not exist"
        assert os.path.exists(self.EXP_COEF_FILE), "EXP coef file does not exist"

        with open(self.EXP_CONFIG_FILE) as fp, chdir(self.EXP_CONFIG_FILE.parent):
            basis = pyEXP.basis.Basis.factory(fp.read())

        coefs = pyEXP.coefs.Coefs.factory(str(self.EXP_COEF_FILE))

        self.potential = PyEXPPotential(
            basis=basis,
            coefs=coefs,
            units=self.exp_units,
        )
        return super().setup_method()


class TestEXPSingle(EXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_SINGLE_COEF_FILE


class TestEXPMulti(EXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_MULTI_COEF_FILE


class TestPyEXPSingle(PyEXPTestBase):
    EXP_CONFIG_FILE = EXP_CONFIG_FILE
    EXP_COEF_FILE = EXP_SINGLE_COEF_FILE


class TestPyEXPMulti(PyEXPTestBase):
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


@pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP")
def test_pyexp_unit_tests():
    """Test PyEXPPotential static/dynamic behavior"""
    units = EXPTestBase.exp_units

    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())

    coefs_single = pyEXP.coefs.Coefs.factory(str(EXP_SINGLE_COEF_FILE))
    coefs_multi = pyEXP.coefs.Coefs.factory(str(EXP_MULTI_COEF_FILE))

    pot_single = PyEXPPotential(basis=basis, coefs=coefs_single, units=units)
    pot_multi = PyEXPPotential(basis=basis, coefs=coefs_multi, units=units)

    assert pot_single.static is True
    assert pot_multi.static is False

    test_x = [8.0, 0, 0] * u.kpc
    assert u.allclose(
        pot_single.energy(test_x, t=0 * u.Gyr),
        pot_single.energy(test_x, t=1.4 * u.Gyr),
    )
    assert not u.allclose(
        pot_multi.energy(test_x, t=0 * u.Gyr),
        pot_multi.energy(test_x, t=1.4 * u.Gyr),
    )

    # check tmin/tmax
    assert u.allclose(pot_multi.tmin_exp, 0.0 * u.Gyr)
    assert u.allclose(pot_multi.tmax_exp, 2.0 * u.Gyr)


def test_multi_different_snapshot_time_unit():
    pot_multi = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_SNAPSHOT_TIME_FILE,
        units=EXPTestBase.exp_units,
        snapshot_time_unit=u.Gyr,
    )
    x = [8.0, 0, 0] * u.kpc
    val0 = pot_multi.energy(x, t=0.0 * u.Gyr)
    val1 = pot_multi.energy(x, t=1.0 * u.Gyr)
    assert np.isclose(val1 / val0, 3.0)  # see: generate_exp.py

    assert u.allclose(pot_multi.tmin_exp, 0.0 * u.Gyr)
    assert u.allclose(pot_multi.tmax_exp, 1.0 * u.Gyr)


@pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP")
def test_pyexp_multi_different_snapshot_time_unit():
    """Test PyEXPPotential with different snapshot time units"""
    units = EXPTestBase.exp_units

    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())

    coefs = pyEXP.coefs.Coefs.factory(str(EXP_MULTI_COEF_SNAPSHOT_TIME_FILE))

    pot_multi = PyEXPPotential(
        basis=basis, coefs=coefs, units=units, snapshot_time_unit=u.Gyr
    )
    x = [8.0, 0, 0] * u.kpc
    val0 = pot_multi.energy(x, t=0.0 * u.Gyr)
    val1 = pot_multi.energy(x, t=1.0 * u.Gyr)
    assert np.isclose(val1 / val0, 3.0)  # see: generate_exp.py

    assert u.allclose(pot_multi.tmin_exp, 0.0 * u.Gyr)
    assert u.allclose(pot_multi.tmax_exp, 1.0 * u.Gyr)


def test_cython_exceptions():
    """Test various exceptions propagated from C++"""
    units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)
    with pytest.raises(RuntimeError, match="file"):
        EXPPotential(
            config_file="nonexistent_config.yml",
            coef_file=EXP_SINGLE_COEF_FILE,
            snapshot_index=0,
            units=units,
        )

    with pytest.raises(RuntimeError, match="index"):
        EXPPotential(
            config_file=EXP_CONFIG_FILE,
            coef_file=EXP_SINGLE_COEF_FILE,
            snapshot_index=0xBAD,
            units=units,
        )

    with pytest.raises(RuntimeError, match="time"):
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
    with pytest.raises(RuntimeError, match="time"):
        pot.energy([0, 0, 0], t=float(0xBAD))

    w0 = gd.PhaseSpacePosition(
        pos=[-8, 0.0, 0.0] * u.kpc,
        vel=[0.0, 220, 0.0] * u.km / u.s,
    )
    with pytest.raises(RuntimeError, match="time"):
        gp.Hamiltonian(pot).integrate_orbit(
            w0, dt=1.0, t1=float(0xBAD), t2=float(0xBADBAD)
        )


@pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP")
def test_pyexp_exceptions():
    """Test various exceptions for PyEXPPotential"""
    units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)

    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())
    coefs = pyEXP.coefs.Coefs.factory(str(EXP_MULTI_COEF_FILE))

    # Test with None
    with pytest.raises(ValueError, match="BiorthBasis"):
        PyEXPPotential(basis=None, coefs=None, units=units)

    # Test with a real Coefs object that is empty
    empty_coefs = pyEXP.coefs.Coefs(type="empty", verbose=False)
    with pytest.raises(RuntimeError, match="Coefs"):
        PyEXPPotential(basis=basis, coefs=empty_coefs, units=units)

    # Test with a non-BiorthBasis
    with open(EXP_FIELD_CONFIG_FILE) as fp, chdir(EXP_FIELD_CONFIG_FILE.parent):
        field_basis = pyEXP.basis.FieldBasis(fp.read())
    with pytest.raises(RuntimeError, match="BiorthBasis"):
        PyEXPPotential(basis=field_basis, coefs=coefs, units=units)

    # Test with valid objects but runtime errors
    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())

    pot = PyEXPPotential(basis=basis, coefs=coefs, units=units)
    with pytest.raises(RuntimeError, match="time"):
        pot.energy([0, 0, 0], t=float(0xBAD))


def _make_exp_pot(config_fn, coef_fn):
    return EXPPotential(
        config_file=config_fn,
        coef_file=coef_fn,
        units=EXP_UNITS,
    )


def _make_pyexp_pot(config_fn, coef_fn):
    return PyEXPPotential(
        basis=_load_pyexp_basis(config_fn),
        coefs=pyEXP.coefs.Coefs.factory(str(coef_fn)),
        units=EXP_UNITS,
    )


def _load_pyexp_basis(config_file):
    """Helper to load pyEXP basis for parametrized tests"""
    if not HAVE_PYEXP:
        return None
    with open(config_file) as fp, chdir(config_file.parent):
        return pyEXP.basis.Basis.factory(fp.read())


potentials_parametrize = pytest.mark.parametrize(
    "make_pot",
    [
        pytest.param(_make_exp_pot, id="exp"),
        pytest.param(
            _make_pyexp_pot,
            id="pyexp",
            marks=pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP"),
        ),
    ],
)


@potentials_parametrize
def test_composite_parametrized(make_pot):
    """Test that both EXPPotential and PyEXPPotential can be used in a CompositePotential"""
    pot_single = make_pot(EXP_CONFIG_FILE, EXP_SINGLE_COEF_FILE)

    pot_multi = make_pot(EXP_CONFIG_FILE, EXP_MULTI_COEF_FILE)
    composite_pot = pot_single + pot_multi
    assert isinstance(
        composite_pot, gp.potential.ccompositepotential.CCompositePotential
    )

    # Test potential energy addition
    test_x = [1.0, 2.0, 3.0] * u.kpc
    assert u.allclose(
        composite_pot.energy(test_x, t=0 * u.Gyr),
        pot_single.energy(test_x, t=0 * u.Gyr) + pot_multi.energy(test_x, t=0 * u.Gyr),
    )
    assert u.allclose(
        composite_pot.energy(test_x, t=1.4 * u.Gyr),
        pot_single.energy(test_x, t=1.4 * u.Gyr)
        + pot_multi.energy(test_x, t=1.4 * u.Gyr),
    )

    # Test gradient addition
    assert u.allclose(
        composite_pot.gradient(test_x, t=0 * u.Gyr),
        pot_single.gradient(test_x, t=0 * u.Gyr)
        + pot_multi.gradient(test_x, t=0 * u.Gyr),
    )
    assert u.allclose(
        composite_pot.gradient(test_x, t=1.4 * u.Gyr),
        pot_single.gradient(test_x, t=1.4 * u.Gyr)
        + pot_multi.gradient(test_x, t=1.4 * u.Gyr),
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


@potentials_parametrize
def test_replace_units(make_pot):
    """Test that replace_units works for both EXPPotential and PyEXPPotential"""
    pot = make_pot(EXP_CONFIG_FILE, EXP_SINGLE_COEF_FILE)

    new_units = SimulationUnitSystem(
        mass=EXP_UNITS["mass"] * 2.0,
        length=EXP_UNITS["length"],
        G=1.0,
    )
    pot_replaced = pot.replace_units(new_units)

    assert pot_replaced.units == new_units
    assert pot_replaced is not pot

    x = [1.0, 2.0, 3.0] * u.kpc
    e1 = pot.energy(x)

    x_new = x.to_value(new_units["length"]) * new_units["length"]
    e2 = pot_replaced.energy(x_new)

    assert u.isclose(e1, e2 / 2.0)


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


def test_replicate():
    """Test that replicate works for EXPPotential"""

    units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)
    pot = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_FILE,
        units=units,
        snapshot_index=0,
    )

    pot_replicated = pot.replicate(snapshot_index=1)

    assert pot_replicated.units == pot.units
    assert pot_replicated.parameters["snapshot_index"] == 1
    assert pot.parameters["snapshot_index"] == 0
    assert pot_replicated is not pot  # should be a new instance

    # Check that the energy at a point is not the same in both instances
    x = [1.0, 2.0, 3.0] * u.kpc
    e1 = pot.energy(x)
    e2 = pot_replicated.energy(x)
    assert not u.isclose(e1, e2)


@pytest.mark.xfail(reason="replicate not supported by PyEXP")
@pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP")
def test_pyexp_replicate():
    """Test that replicate works for PyEXPPotential using coef_file"""

    units = SimulationUnitSystem(mass=1e11 * u.Msun, length=2.5 * u.kpc, G=1)
    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())
    coefs = pyEXP.coefs.Coefs.factory(str(EXP_SINGLE_COEF_FILE))

    pot = PyEXPPotential(
        basis=basis,
        coefs=coefs,
        units=units,
    )

    pot_replicated = pot.replicate(coef_file=str(EXP_MULTI_COEF_FILE))

    assert pot_replicated.units == pot.units
    assert Path(pot_replicated.parameters["coef_file"]) == Path(EXP_MULTI_COEF_FILE)
    assert Path(pot.parameters["coef_file"]) == Path(EXP_SINGLE_COEF_FILE)
    assert pot_replicated is not pot  # should be a new instance

    x = [1.0, 2.0, 3.0] * u.kpc
    e1 = pot.energy(x, t=0 * u.Gyr)
    e2 = pot_replicated.energy(x, t=0 * u.Gyr)
    assert not u.isclose(e1, e2)


@pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP")
def test_exp_pyexp_consistency_single():
    """Test that EXPPotential and PyEXPPotential give the same results"""

    # Create EXPPotential
    exp_pot = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_SINGLE_COEF_FILE,
        units=EXP_UNITS,
    )

    # Create PyEXPPotential with same data
    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())
    coefs = pyEXP.coefs.Coefs.factory(str(EXP_SINGLE_COEF_FILE))
    pyexp_pot = PyEXPPotential(basis=basis, coefs=coefs, units=EXP_UNITS)

    x = [1.0, 2.0, 3.0] * u.kpc

    # Compare energy
    exp_energy = exp_pot.energy(x)
    pyexp_energy = pyexp_pot.energy(x)
    assert u.allclose(exp_energy, pyexp_energy)

    # Compare density
    exp_density = exp_pot.density(x)
    pyexp_density = pyexp_pot.density(x)
    assert u.allclose(exp_density, pyexp_density)

    # Compare gradient
    exp_gradient = exp_pot.gradient(x)
    pyexp_gradient = pyexp_pot.gradient(x)
    assert u.allclose(exp_gradient, pyexp_gradient)


@pytest.mark.skipif(not HAVE_PYEXP, reason="requires pyEXP")
def test_exp_pyexp_consistency_multi():
    """Test time-dependent consistency between EXPPotential and PyEXPPotential."""
    exp_dynamic = EXPPotential(
        config_file=EXP_CONFIG_FILE,
        coef_file=EXP_MULTI_COEF_FILE,
        units=EXP_UNITS,
    )

    with open(EXP_CONFIG_FILE) as fp, chdir(EXP_CONFIG_FILE.parent):
        basis = pyEXP.basis.Basis.factory(fp.read())
    coefs = pyEXP.coefs.Coefs.factory(str(EXP_MULTI_COEF_FILE))

    pyexp_dynamic = PyEXPPotential(
        basis=basis,
        coefs=coefs,
        units=EXP_UNITS,
    )

    assert exp_dynamic.static is False
    assert pyexp_dynamic.static is False

    x = [2.5, -1.5, 0.4] * u.kpc
    times = [0.0, 1.4] * u.Gyr

    for t in times:
        exp_energy = exp_dynamic.energy(x, t=t)
        pyexp_energy = pyexp_dynamic.energy(x, t=t)
        exp_density = exp_dynamic.density(x, t=t)
        pyexp_density = pyexp_dynamic.density(x, t=t)
        exp_gradient = exp_dynamic.gradient(x, t=t)
        pyexp_gradient = pyexp_dynamic.gradient(x, t=t)

        assert u.allclose(exp_energy, pyexp_energy)
        assert u.allclose(exp_density, pyexp_density)
        assert u.allclose(exp_gradient, pyexp_gradient)
