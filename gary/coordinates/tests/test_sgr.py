# coding: utf-8
"""
    Test the coordinates class that represents the plane of orbit of the Sgr dwarf galaxy.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# This package
from ..sgr import *

def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree),
                   coord.Angle(-11.4351, u.degree))
    c.transform_to(Sagittarius)

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(Sagittarius)

    c = Sagittarius(coord.Angle(217.2141, u.degree),
                    coord.Angle(-11.4351, u.degree))
    c.transform_to(coord.ICRS)
    c.transform_to(coord.Galactic)

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    s = c.transform_to(Sagittarius)

    # with distance
    c = Sagittarius(coord.Angle(217.2141, u.degree),
                    coord.Angle(-11.4351, u.degree),
                    distance=15*u.kpc)
    c.transform_to(coord.ICRS)
    c2 = c.transform_to(coord.Galactic)
    assert np.allclose(c2.distance.value, c.distance.value)

def test_against_David_Law():
    """ Test my code against an output file from using David Law's cpp code. Do:

            g++ SgrCoord.cpp; ./a.out

        to generate the data file, SgrCoord_data.

    """
    filename = get_pkg_data_filename('SgrCoord_data')
    law_data = np.genfromtxt(filename, names=True, delimiter=',')

    c = coord.Galactic(law_data["l"]*u.deg, law_data["b"]*u.deg)
    sgr_coords = c.transform_to(Sagittarius)

    law_sgr_coords = Sagittarius(Lambda=law_data["lambda"]*u.deg, Beta=law_data["beta"]*u.deg)

    sep = sgr_coords.separation(law_sgr_coords).arcsec*u.arcsec
    assert np.all(sep < 1.*u.arcsec)
