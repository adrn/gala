# coding: utf-8

""" test reading/writing potentials to files """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# Project
from ..io import load, save
from ..core import CompositePotential
from ..builtin import IsochronePotential, KeplerPotential
from ..builtin.special import PW14Potential
from ...units import galactic

def test_read():
    potential = load(get_pkg_data_filename('Plummer.yml'))
    assert np.allclose(potential.parameters['m'], 100000000000.)
    assert np.allclose(potential.parameters['b'], 0.26)

    potential = load(get_pkg_data_filename('HarmonicOscillator1D.yml'))
    assert potential.units is None

    potential = load(get_pkg_data_filename('Composite.yml'))
    assert 'disk' in potential.keys()
    assert 'halo' in potential.keys()
    assert str(potential) == "CompositePotential"

def test_write():

    tmp_filename = "/tmp/potential.yml"

    # try a simple potential
    potential = IsochronePotential(m=1E11, b=0.76, units=galactic)

    with open(tmp_filename,'w') as f:
        save(potential, f)

    save(potential, tmp_filename)

    # more complex
    potential = PW14Potential()

    with open(tmp_filename,'w') as f:
        save(potential, f)

    save(potential, tmp_filename)

    # composite potential
    potential = CompositePotential(halo=KeplerPotential(m=1E11, units=galactic),
                                   bulge=IsochronePotential(m=1E11, b=0.76, units=galactic))
    save(potential, tmp_filename)

def test_units():
    import astropy.units as u

    tmp_filename = "/tmp/potential.yml"

    # try a simple potential
    potential = KeplerPotential(m=1E11, units=[u.kpc,u.Gyr,u.Msun,u.radian])
    save(potential, tmp_filename)
    p = load(tmp_filename)
