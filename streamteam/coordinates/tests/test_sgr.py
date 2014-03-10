# coding: utf-8
"""
    Test the coordinates class that represents the plane of orbit of the Sgr dwarf galaxy.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from ..sgr import *

this_path = os.path.split(__file__)[0]
law_data = np.genfromtxt(os.path.join(this_path, "SgrCoord_data"),
                         names=True, delimiter=',')

def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree),
                   coord.Angle(-11.4351, u.degree))
    c.transform_to(SgrCoordinates)

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(SgrCoordinates)

    c = SgrCoordinates(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(coord.ICRS)
    c.transform_to(coord.Galactic)

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    assert SgrCoordinates(c).Lambda.radian == \
           c.transform_to(SgrCoordinates).Lambda.radian

def test_against_David_Law():
    """ Test my code against an output file from using David Law's cpp code. Do:

            g++ SgrCoord.cpp; ./a.out

        to generate the data file, SgrCoord_data.

    """

    c = coord.Galactic(law_data["l"], law_data["b"], unit=(u.degree,u.degree))
    sgr_coords = c.transform_to(SgrCoordinates)

    law_sgr_coords = SgrCoordinates(law_data["lambda"], law_data["beta"],
                                    unit=(u.degree, u.degree))

    sep = sgr_coords.separation(law_sgr_coords).arcsec*u.arcsec
    assert np.all(sep < 1.*u.arcsec)

def test_distance_to_sgr_plane():
    N = 100
    ra = coord.Angle(np.random.uniform(0, 360., size=N)*u.degree)
    dec = coord.Angle(np.random.uniform(-90, 90., size=N)*u.degree)
    distance = np.random.uniform(10, 60., size=N)*u.kpc

    D = distance_to_sgr_plane(ra, dec, distance)
    assert hasattr(D, "unit")
