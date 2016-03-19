# coding: utf-8

""" test reading/writing potentials to files """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# Project
from ..io import load, save
from ..core import CompositePotential
from ..builtin import IsochronePotential, KeplerPotential
from ..builtin.special import PW14Potential
from ...units import galactic

def test_read_plummer():
    potential = load(get_pkg_data_filename('Plummer.yml'))
    assert np.allclose(potential.parameters['m'].value, 100000000000.)
    assert np.allclose(potential.parameters['b'].value, 0.26)
    assert potential.parameters['b'].unit == u.kpc

def test_read_harmonic_oscillator():
    potential = load(get_pkg_data_filename('HarmonicOscillator1D.yml'))
    assert potential.units is None

def test_read_composite():
    potential = load(get_pkg_data_filename('Composite.yml'))
    assert '0' in potential.keys()
    assert 'disk' in potential.keys()
    assert str(potential) == "CompositePotential"

def test_write_isochrone(tmpdir):
    tmp_filename = tmpdir.join("potential.yml")

    # try a simple potential
    potential = IsochronePotential(m=1E11, b=0.76, units=galactic)

    with open(tmp_filename,'w') as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)

def test_write_isochrone_units(tmpdir):
    tmp_filename = tmpdir.join("potential.yml")

    # try a simple potential with units
    potential = IsochronePotential(m=1E11*u.Msun, b=0.76*u.kpc, units=galactic)

    with open(tmp_filename,'w') as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)

def test_write_pw14(tmpdir):
    tmp_filename = tmpdir.join("potential.yml")

    # more complex
    potential = PW14Potential()

    with open(tmp_filename,'w') as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)

def test_write_composite(tmpdir):
    tmp_filename = tmpdir.join("potential.yml")

    # composite potential
    potential = CompositePotential(halo=KeplerPotential(m=1E11, units=galactic),
                                   bulge=IsochronePotential(m=1E11, b=0.76, units=galactic))
    save(potential, tmp_filename)
    p = load(tmp_filename)

def test_units(tmpdir):
    import astropy.units as u

    tmp_filename = tmpdir.join("potential.yml")

    # try a simple potential
    potential = KeplerPotential(m=1E11, units=[u.kpc,u.Gyr,u.Msun,u.radian])
    save(potential, tmp_filename)
    p = load(tmp_filename)
