"""
Test the coordinates class that represents the plane of orbit of the Sgr dwarf galaxy.
"""

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# This package
from ..sgr import SagittariusLaw10, SagittariusVasiliev21


def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree))
    c.transform_to(SagittariusLaw10())

    c = coord.Galactic(coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree))
    c.transform_to(SagittariusLaw10())

    c = SagittariusLaw10(
        coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree)
    )
    c.transform_to(coord.ICRS())
    c.transform_to(coord.Galactic())

    c = coord.Galactic(coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree))
    c.transform_to(SagittariusLaw10())

    # with distance
    c = SagittariusLaw10(
        coord.Angle(217.2141, u.degree),
        coord.Angle(-11.4351, u.degree),
        distance=15 * u.kpc,
    )
    c.transform_to(coord.ICRS())
    c2 = c.transform_to(coord.Galactic())
    assert np.allclose(c2.distance.value, c.distance.value)


def test_against_David_Law():
    """Test my code against an output file from using David Law's cpp code. Do:

        g++ SgrCoord.cpp; ./a.out

    to generate the data file, SgrCoord_data.

    """
    filename = get_pkg_data_filename("SgrCoord_data")
    law_data = np.genfromtxt(filename, names=True, delimiter=",")

    c = coord.Galactic(law_data["l"] * u.deg, law_data["b"] * u.deg)
    sgr_coords = c.transform_to(SagittariusLaw10())

    law_sgr_coords = SagittariusLaw10(
        Lambda=law_data["lambda"] * u.deg, Beta=law_data["beta"] * u.deg
    )

    sep = sgr_coords.separation(law_sgr_coords).arcsec * u.arcsec
    assert np.all(sep < 1.0 * u.arcsec)


def test_v21():
    filename = get_pkg_data_filename("Vasiliev2020-Sagittarius-subset.csv")
    test_data = at.Table.read(filename, format="ascii.csv")

    c = coord.SkyCoord(test_data["ra"] * u.deg, test_data["dec"] * u.deg)
    sgr_c = c.transform_to(SagittariusVasiliev21())

    assert np.allclose(sgr_c.Lambda.degree, test_data["Lambda"], atol=1e-3)
    assert np.allclose(sgr_c.Beta.degree, test_data["Beta"], atol=1e-3)
