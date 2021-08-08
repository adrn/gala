"""
    Test the coordinates class that represents the plane of orbit of the Sgr dwarf galaxy.
"""

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# This package
from gala.util import GalaDeprecationWarning
from ..sgr import SagittariusLaw10, Sagittarius


def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree),
                   coord.Angle(-11.4351, u.degree))
    c.transform_to(SagittariusLaw10())

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(SagittariusLaw10())

    c = SagittariusLaw10(coord.Angle(217.2141, u.degree),
                         coord.Angle(-11.4351, u.degree))
    c.transform_to(coord.ICRS())
    c.transform_to(coord.Galactic())

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(SagittariusLaw10())

    # with distance
    c = SagittariusLaw10(coord.Angle(217.2141, u.degree),
                         coord.Angle(-11.4351, u.degree),
                         distance=15*u.kpc)
    c.transform_to(coord.ICRS())
    c2 = c.transform_to(coord.Galactic())
    assert np.allclose(c2.distance.value, c.distance.value)

    # TODO: remove this in next version
    # For now: make sure old class still works
    from astropy.tests.helper import catch_warnings
    with catch_warnings(GalaDeprecationWarning) as w:
        c = Sagittarius(217.2141*u.degree, -11.4351*u.degree)
    assert len(w) > 0
    c2 = c.transform_to(coord.Galactic())
    c3 = c2.transform_to(Sagittarius())
    assert np.allclose(c3.Lambda.degree, c.Lambda.degree)
    assert np.allclose(c3.Beta.degree, c.Beta.degree)


def test_against_David_Law():
    """ Test my code against an output file from using David Law's cpp code. Do:

            g++ SgrCoord.cpp; ./a.out

        to generate the data file, SgrCoord_data.

    """
    filename = get_pkg_data_filename('SgrCoord_data')
    law_data = np.genfromtxt(filename, names=True, delimiter=',')

    c = coord.Galactic(law_data["l"]*u.deg, law_data["b"]*u.deg)
    sgr_coords = c.transform_to(SagittariusLaw10())

    law_sgr_coords = SagittariusLaw10(Lambda=law_data["lambda"]*u.deg,
                                      Beta=law_data["beta"]*u.deg)

    sep = sgr_coords.separation(law_sgr_coords).arcsec*u.arcsec
    assert np.all(sep < 1.*u.arcsec)
