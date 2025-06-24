import astropy.coordinates as coord
import astropy.units as u
import numpy as np

from ..pal5 import Pal5PriceWhelan18


def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree))
    c.transform_to(Pal5PriceWhelan18())

    c = coord.Galactic(coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree))
    c.transform_to(Pal5PriceWhelan18())

    c = Pal5PriceWhelan18(217.2141 * u.degree, -11.4351 * u.degree)
    c.transform_to(coord.ICRS())
    c.transform_to(coord.Galactic())

    c = coord.Galactic(coord.Angle(217.2141, u.degree), coord.Angle(-11.4351, u.degree))
    c.transform_to(Pal5PriceWhelan18())

    # with distance
    c = Pal5PriceWhelan18(
        coord.Angle(217.2141, u.degree),
        coord.Angle(-11.4351, u.degree),
        distance=15 * u.kpc,
    )
    c.transform_to(coord.ICRS())
    c2 = c.transform_to(coord.Galactic())
    assert np.allclose(c2.distance.value, c.distance.value)
