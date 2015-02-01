# coding: utf-8

""" test reading/writing potentials to files """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

# Project
from ..io import read, write
from ..core import CompositePotential
from ..builtin import IsochronePotential, PointMassPotential
from ..custom import PW14Potential
from ...units import galactic

# TODO: config item to specify path to test data?
test_data_path = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                 "../../../test-data/"))

def test_read():
    f1 = os.path.join(test_data_path, 'potential', 'isochrone.yml')
    potential = read(f1)
    assert np.allclose(potential.parameters['m'], 1E11)
    assert np.allclose(potential.parameters['b'], 0.76)

    f2 = os.path.join(test_data_path, 'potential', 'pw14.yml')
    potential = read(f2)

    f3 = os.path.join(test_data_path, 'potential', 'pw14_2.yml')
    potential = read(f3)

    f4 = os.path.join(test_data_path, 'potential', 'composite.yml')
    potential = read(f4)
    assert str(potential) == "CompositePotential"

def test_write():

    tmp_filename = "/tmp/potential.yml"

    # try a simple potential
    potential = IsochronePotential(m=1E11, b=0.76, units=galactic)

    with open(tmp_filename,'w') as f:
        write(potential, f)

    write(potential, tmp_filename)

    # more complex
    potential = PW14Potential()

    with open(tmp_filename,'w') as f:
        write(potential, f)

    write(potential, tmp_filename)

    # composite potential
    potential = CompositePotential(halo=PointMassPotential(m=1E11, x0=0., units=galactic),
                                   bulge=IsochronePotential(m=1E11, b=0.76, units=galactic))
    write(potential, tmp_filename)

