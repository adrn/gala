"""test reading/writing potentials to files"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_filename

from gala._cconfig import GSL_ENABLED

from ....units import DimensionlessUnitSystem, galactic
from ...scf import SCFPotential
from ..builtin import IsochronePotential, KeplerPotential
from ..builtin.special import LM10Potential
from ..ccompositepotential import CCompositePotential
from ..core import CompositePotential
from ..io import load, save


def test_read_plummer():
    potential = load(get_pkg_data_filename("Plummer.yml"))
    assert np.allclose(potential.parameters["m"].value, 100000000000.0)
    assert np.allclose(potential.parameters["b"].value, 0.26)
    assert potential.parameters["b"].unit == u.kpc


def test_read_harmonic_oscillator():
    potential = load(get_pkg_data_filename("HarmonicOscillator1D.yml"))
    assert isinstance(potential.units, DimensionlessUnitSystem)


def test_read_composite():
    potential = load(get_pkg_data_filename("Composite.yml"))
    assert "halo" in potential
    assert "disk" in potential
    assert str(potential) == "CompositePotential"
    assert potential.units["length"] == u.kpc
    assert potential.units["speed"] == u.km / u.s


def test_read_lm10():
    potential = load(get_pkg_data_filename("lm10.yml"))
    assert "halo" in potential
    assert "disk" in potential
    assert str(potential) == "LM10Potential"
    assert np.allclose(potential["disk"].parameters["a"].value, 10)
    assert np.allclose(potential["disk"].parameters["b"].value, 0.26)
    assert np.allclose(potential["disk"].parameters["m"].value, 150000.0)


def test_write_isochrone(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a simple potential
    potential = IsochronePotential(m=1e11, b=0.76, units=galactic)

    with open(tmp_filename, "w", encoding="utf-8") as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_write_isochrone_units(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a simple potential with units
    potential = IsochronePotential(m=1e11 * u.Msun, b=0.76 * u.kpc, units=galactic)

    with open(tmp_filename, "w", encoding="utf-8") as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_write_lm10(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # more complex
    potential = LM10Potential(disk={"m": 5e12 * u.Msun})
    potential_default = LM10Potential()
    v1 = potential.energy([4.0, 0, 0])
    v2 = potential_default.energy([4.0, 0, 0])

    with open(tmp_filename, "w", encoding="utf-8") as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)
    assert u.allclose(p["disk"].parameters["m"], 5e12 * u.Msun)
    assert u.allclose(v1, p.energy([4.0, 0, 0]))
    assert not u.allclose(v2, p.energy([4.0, 0, 0]))


def test_write_composite(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))
    print(tmp_filename)

    # composite potential
    potential = CompositePotential(
        halo=KeplerPotential(m=1e11, units=galactic),
        bulge=IsochronePotential(m=1e11, b=0.76, units=galactic),
    )
    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_write_ccomposite(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # composite potential
    potential = CCompositePotential(
        halo=KeplerPotential(m=1e11, units=galactic),
        bulge=IsochronePotential(m=1e11, b=0.76, units=galactic),
    )
    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_units(tmpdir):
    import astropy.units as u

    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a simple potential
    potential = KeplerPotential(m=1e11, units=[u.kpc, u.Gyr, u.Msun, u.radian])
    save(potential, tmp_filename)
    p = load(tmp_filename)


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
def test_read_write_SCF(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a basic SCF potential
    potential = SCFPotential(100, 1, np.zeros((4, 3, 2)), np.zeros((4, 3, 2)))
    save(potential, tmp_filename)
    p = load(tmp_filename)
