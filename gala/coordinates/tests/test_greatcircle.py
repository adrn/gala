# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# This project
from ..greatcircle import GreatCircleICRSFrame

def test_cls_init():
    pole = coord.SkyCoord(ra=72.2643*u.deg, dec=-20.6575*u.deg)
    GreatCircleICRSFrame(pole=pole)
    GreatCircleICRSFrame(pole=pole, ra0=160*u.deg)


def test_transform_against_koposov():
    from .helpers import sphere_rotate

    # ra, dec, ra0
    pole_ra0s = [(71.241, -18.941, 0.),
                 (151.441, 81.193, 71.4123),
                 (210.412, 7.134, 200.)]

    ra = np.random.uniform(0, 360, 128)
    dec = np.random.uniform(-90, 90, len(ra))
    c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    for pole_ra, pole_dec, ra0 in pole_ra0s:
        kop = sphere_rotate(ra, dec, pole_ra, pole_dec, ra0)

        pole = coord.SkyCoord(ra=pole_ra*u.deg,
                              dec=pole_dec*u.deg)
        fr = GreatCircleICRSFrame(pole=pole, ra0=ra0*u.deg)
        apw = c.transform_to(fr)

        phi1 = apw.phi1.wrap_at(180*u.deg).degree
        assert np.allclose(kop[0], phi1)
        assert np.allclose(kop[1], apw.phi2.degree)
