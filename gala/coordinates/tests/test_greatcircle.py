# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# This project
from ..greatcircle import (GreatCircleICRSFrame, make_greatcircle_cls,
                           pole_from_endpoints, sph_midpoint)


def test_cls_init():
    pole = coord.SkyCoord(ra=72.2643*u.deg, dec=-20.6575*u.deg)
    GreatCircleICRSFrame(pole=pole)
    GreatCircleICRSFrame(pole=pole, ra0=160*u.deg)

    points = coord.SkyCoord(ra=[-38.8, 4.7]*u.deg, dec=[-45.1, -51.7]*u.deg)
    fr = GreatCircleICRSFrame.from_endpoints(points[0], points[1])
    assert u.allclose(fr.pole.ra, 359.1*u.deg, atol=1e-1*u.deg)
    assert u.allclose(fr.pole.dec, 38.2*u.deg, atol=1e-1*u.deg)

    fr = GreatCircleICRSFrame.from_endpoints(points[0], points[1],
                                             ra0=100*u.deg)

    fr = GreatCircleICRSFrame.from_endpoints(points[0], points[1],
                                             rotation=100*u.deg)

    with pytest.raises(ValueError):
        GreatCircleICRSFrame(pole=pole, ra0=160*u.deg,
                             center=pole)


def test_init_center():
    galcen = coord.Galactocentric()
    stupid_gal = GreatCircleICRSFrame(
        pole=coord.Galactic._ngp_J2000.transform_to(coord.ICRS()),
        center=galcen.galcen_coord)
    gal = coord.Galactic(50*u.deg, 20*u.deg)
    gal2 = gal.transform_to(stupid_gal)

    assert np.isclose(gal.l.degree, gal2.phi1.degree)
    assert np.isclose(gal.b.degree, gal2.phi2.degree)


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

    # Test with no RA zero point
    fr = GreatCircleICRSFrame(pole=pole)
    apw = c.transform_to(fr)
    assert np.isfinite(apw.phi1).all()
    assert np.isfinite(apw.phi2).all()


def test_make_function():
    pole = coord.SkyCoord(ra=72.2643*u.deg, dec=-20.6575*u.deg)

    kwargs = [dict(pole=pole),
              dict(pole=pole, ra0=100*u.deg),
              dict(pole=pole, rotation=50*u.deg),
              dict(pole=pole, ra0=100*u.deg, rotation=50*u.deg)]
    for kw in kwargs:
        cls = make_greatcircle_cls('Michael', 'This is the docstring header',
                                   **kw)
        fr = cls(phi1=100*u.deg, phi2=10*u.deg)
        fr.transform_to(coord.ICRS())


def test_pole_from_endpoints():
    c1 = coord.SkyCoord(0*u.deg, 0*u.deg)
    c2 = coord.SkyCoord(90*u.deg, 0*u.deg)
    pole = pole_from_endpoints(c1, c2)
    assert u.allclose(pole.dec, 90*u.deg)

    c1 = coord.SkyCoord(0*u.deg, 0*u.deg)
    c2 = coord.SkyCoord(0*u.deg, 90*u.deg)
    pole = pole_from_endpoints(c1, c2)
    assert u.allclose(pole.ra, 270*u.deg)
    assert u.allclose(pole.dec, 0*u.deg)

    # Should work even if coord has velocities:
    c1 = coord.SkyCoord(0*u.deg, 0*u.deg,
                        pm_ra_cosdec=10*u.mas/u.yr,
                        pm_dec=-0.5*u.mas/u.yr)
    c2 = coord.SkyCoord(0*u.deg, 90*u.deg,
                        pm_ra_cosdec=10*u.mas/u.yr,
                        pm_dec=-0.5*u.mas/u.yr)
    pole = pole_from_endpoints(c1, c2)
    assert u.allclose(pole.ra, 270*u.deg)
    assert u.allclose(pole.dec, 0*u.deg)


def test_pole_from_xyz():
    xnew = coord.UnitSphericalRepresentation(185*u.deg, 32.5*u.deg).to_cartesian()
    ynew = coord.UnitSphericalRepresentation(275*u.deg, 0*u.deg).to_cartesian()
    znew = xnew.cross(ynew)

    fr1 = GreatCircleICRSFrame.from_xyz(xnew, ynew, znew)
    fr2 = GreatCircleICRSFrame.from_xyz(xnew, ynew)
    fr3 = GreatCircleICRSFrame.from_xyz(xnew, znew=znew)
    fr4 = GreatCircleICRSFrame.from_xyz(ynew=ynew, znew=znew)

    for fr in [fr2, fr3, fr4]:
        assert np.isclose(fr1.pole.ra.degree, fr.pole.ra.degree)
        assert np.isclose(fr1.pole.dec.degree, fr.pole.dec.degree)
        assert np.isclose(fr1.center.ra.degree, fr.center.ra.degree)
        assert np.isclose(fr1.center.dec.degree, fr.center.dec.degree)

    with pytest.raises(ValueError):
        GreatCircleICRSFrame.from_xyz(xnew)


def test_sph_midpoint():
    c1 = coord.SkyCoord(0*u.deg, 0*u.deg)
    c2 = coord.SkyCoord(90*u.deg, 0*u.deg)
    midpt = sph_midpoint(c1, c2)
    assert u.allclose(midpt.ra, 45*u.deg)
    assert u.allclose(midpt.dec, 0*u.deg)

    c1 = coord.SkyCoord(0*u.deg, 0*u.deg)
    c2 = coord.SkyCoord(0*u.deg, 90*u.deg)
    midpt = sph_midpoint(c1, c2)
    assert u.allclose(midpt.ra, 0*u.deg)
    assert u.allclose(midpt.dec, 45*u.deg)


def test_pole_separation90():
    # Regression test for issue #160
    from astropy.tests.helper import catch_warnings

    for dec in [19.8, 0, -41.3]:  # random values, but 0 is an important test
        pole = coord.SkyCoord(ra=32.5, dec=dec, unit='deg')
        kwargs = [(dict(pole=pole), None),
                  (dict(pole=pole, ra0=100*u.deg), RuntimeWarning),
                  (dict(pole=pole, rotation=50*u.deg), None),
                  (dict(pole=pole, ra0=100*u.deg, rotation=50*u.deg),
                   RuntimeWarning)]

        for kw, warning in kwargs:
            gcfr = GreatCircleICRSFrame(**kw)
            gc = coord.SkyCoord(phi1=np.linspace(0, 360, 100),
                                phi2=0,
                                unit='deg', frame=gcfr)
            with catch_warnings(RuntimeWarning) as w:
                gc = gc.transform_to(coord.ICRS())
            if warning is not None and dec == 0:
                assert len(w) > 0

            assert u.allclose(gc.separation(pole), 90*u.deg)


def test_init_R():
    from ..gd1 import R as gd1_R, GD1Koposov10

    N = 128
    rnd = np.random.RandomState(42)

    gd1_gc_frame = GreatCircleICRSFrame.from_R(gd1_R)
    tmp_in = GD1Koposov10(phi1=rnd.uniform(0, 360, N)*u.deg,
                          phi2=rnd.uniform(-90, 90, N)*u.deg)

    tmp_out = tmp_in.transform_to(gd1_gc_frame)

    assert u.allclose(tmp_in.phi1, tmp_out.phi1)
    assert u.allclose(tmp_in.phi2, tmp_out.phi2)
