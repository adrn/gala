# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# This package
from ..jhelum import JhelumBonaca19


def test_simple():
    c = coord.ICRS(coord.Angle(217.2141, u.degree),
                   coord.Angle(-11.4351, u.degree))
    c.transform_to(JhelumBonaca19())

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(JhelumBonaca19())

    c = JhelumBonaca19(217.2141*u.degree, -11.4351*u.degree)
    c.transform_to(coord.ICRS())
    c.transform_to(coord.Galactic())

    c = coord.Galactic(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(JhelumBonaca19())

    # with distance
    c = JhelumBonaca19(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree),
                       distance=15*u.kpc)
    c.transform_to(coord.ICRS())
    c2 = c.transform_to(coord.Galactic())
    assert np.allclose(c2.distance.value, c.distance.value)
