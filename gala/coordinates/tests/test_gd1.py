# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# This package
from gala.util import GalaDeprecationWarning
from ..gd1 import GD1Koposov10, GD1


def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree),
                   coord.Angle(-11.4351, u.degree))
    c.transform_to(GD1Koposov10())

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(GD1Koposov10())

    c = GD1Koposov10(217.2141*u.degree, -11.4351*u.degree)
    c.transform_to(coord.ICRS)
    c.transform_to(coord.Galactic)

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(GD1Koposov10())

    # with distance
    c = GD1Koposov10(coord.Angle(217.2141, u.degree),
                     coord.Angle(-11.4351, u.degree),
                     distance=15*u.kpc)
    c.transform_to(coord.ICRS)
    c2 = c.transform_to(coord.Galactic)
    assert np.allclose(c2.distance.value, c.distance.value)

    # TODO: remove this in next version
    # For now: make sure old class still works
    from astropy.tests.helper import catch_warnings
    with catch_warnings(GalaDeprecationWarning) as w:
        c = GD1(217.2141*u.degree, -11.4351*u.degree)
    assert len(w) > 0
    c2 = c.transform_to(coord.Galactic)
    c3 = c2.transform_to(GD1())
    assert np.allclose(c3.phi1.degree, c.phi1.degree)
    assert np.allclose(c3.phi2.degree, c.phi2.degree)


def test_koposov():
    # Compare against Table 1 in Koposov et al. 2010

    filename = get_pkg_data_filename('gd1_coord.txt')
    k10_data = np.genfromtxt(filename, names=True, dtype=None)

    k10_icrs = coord.SkyCoord(ra=k10_data['ra'].astype(str),
                              dec=k10_data['dec'].astype(str),
                              unit=(u.hourangle, u.degree))

    k10_gd1 = GD1Koposov10(phi1=k10_data['phi1']*u.degree,
                           phi2=k10_data['phi2']*u.degree)

    gala_gd1 = k10_icrs.transform_to(GD1Koposov10())

    # TODO: why are these so different from the values in Koposov?
    assert np.allclose(k10_gd1.phi1.degree, gala_gd1.phi1.degree,
                       atol=1E-1)
    assert np.allclose(k10_gd1.phi2.degree, gala_gd1.phi2.degree,
                       atol=0.2)
