# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# This project
from ..greatcircle import (
    GreatCircleICRSFrame,
    make_greatcircle_cls,
    pole_from_endpoints,
    sph_midpoint,
)

rng = np.random.default_rng(seed=42)
rand_lon = rng.uniform(0, 2 * np.pi, 15) * u.rad
rand_lat = np.arcsin(rng.uniform(-1, 1, size=15)) * u.rad
poles = [
    coord.SkyCoord(ra=0 * u.deg, dec=90 * u.deg),
    coord.SkyCoord(ra=0 * u.deg, dec=-90 * u.deg),
    coord.SkyCoord(ra=12.3 * u.deg, dec=45.6 * u.deg, distance=1 * u.kpc),
] + [coord.SkyCoord(lon, lat) for lon, lat in zip(rand_lon, rand_lat)]


def get_random_orthogonal(skycoord, rng):
    zhat = np.squeeze((skycoord.cartesian / skycoord.cartesian.norm()).xyz)

    # Random vector orthogonal to the pole:
    x = rng.uniform(size=3)
    x /= np.linalg.norm(x)
    xhat = x - (x @ zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    return coord.SkyCoord(coord.CartesianRepresentation(xhat), frame=skycoord.frame)


@pytest.mark.parametrize("pole", poles)
def test_init_cls(pole):
    origin = get_random_orthogonal(pole, rng)

    GreatCircleICRSFrame(pole=pole, origin=origin)
    GreatCircleICRSFrame(pole=pole, origin=origin, priority="pole")

    with pytest.raises(ValueError):
        GreatCircleICRSFrame(pole=pole, ra0=origin.ra)

    # Slightly adjust the origin so it is not orthogonal:
    new_origin = origin.spherical_offsets_by(
        1.23 * u.deg, -2.42 * u.deg
    )  # random values

    with pytest.warns():
        f1 = GreatCircleICRSFrame(pole=pole, origin=new_origin)

    # default priority="origin"
    assert f1.origin.ra == new_origin.ra
    assert f1.origin.dec == new_origin.dec
    assert np.isclose(f1.origin.cartesian.xyz @ f1.pole.cartesian.xyz, 0.0)

    with pytest.warns():
        f2 = GreatCircleICRSFrame(pole=pole, origin=new_origin, priority="pole")

    assert f2.pole.ra == pole.ra
    assert f2.pole.dec == pole.dec
    assert np.isclose(f2.origin.cartesian.xyz @ f2.pole.cartesian.xyz, 0.0)


@pytest.mark.parametrize("pole", poles)
def test_init_from_pole_ra0(pole):
    GreatCircleICRSFrame.from_pole_ra0(pole, ra0=153 * u.deg)

    disamb = coord.SkyCoord(ra=210 * u.deg, dec=-17 * u.deg)
    GreatCircleICRSFrame.from_pole_ra0(
        pole, ra0=153 * u.deg, origin_disambiguate=disamb
    )


fail_poles = [
    coord.SkyCoord(ra=90 * u.deg, dec=0 * u.deg),
    coord.SkyCoord(ra=13.5399 * u.deg, dec=0 * u.deg),
]


@pytest.mark.parametrize("pole", fail_poles)
def test_init_from_pole_ra0_fail(pole):
    with pytest.raises(ValueError):
        test_init_from_pole_ra0(pole)


@pytest.mark.parametrize("c1", poles)
def test_init_from_endpoints(c1):
    # Random vector for other endpoint:
    x = rng.uniform(size=3)
    x /= np.linalg.norm(x)
    c2 = coord.SkyCoord(coord.CartesianRepresentation(x))

    midpt = coord.SkyCoord(sph_midpoint(c1.squeeze(), c2))
    origin_off = midpt.spherical_offsets_by(1.423 * u.deg, -2.182 * u.deg)

    f1 = GreatCircleICRSFrame.from_endpoints(c1, c2)
    f2 = GreatCircleICRSFrame.from_endpoints(c1, c2, origin=midpt)
    with pytest.warns():
        f3 = GreatCircleICRSFrame.from_endpoints(c1, c2, origin=origin_off)
    assert u.isclose(f3.origin.ra, origin_off.ra)
    assert u.isclose(f3.origin.dec, origin_off.dec)

    if np.abs(c1.dec) != 90 * u.deg:
        f4 = GreatCircleICRSFrame.from_endpoints(c1, c2, ra0=origin_off.ra)

    with pytest.warns():
        f5 = GreatCircleICRSFrame.from_endpoints(
            c1, c2, origin=origin_off, priority="pole"
        )
    assert u.isclose(f5.pole.ra, f1.pole.ra)
    assert u.isclose(f5.pole.dec, f1.pole.dec)


@pytest.mark.parametrize("pole", poles)
def test_make_function(pole):
    origin = get_random_orthogonal(pole, rng)

    cls = make_greatcircle_cls(
        "Michael", "This is the docstring header", pole=pole, origin=origin
    )
    fr = cls(phi1=100 * u.deg, phi2=10 * u.deg)
    fr.transform_to(coord.ICRS())


def test_pole_from_endpoints():
    c1 = coord.SkyCoord(0 * u.deg, 0 * u.deg)
    c2 = coord.SkyCoord(90 * u.deg, 0 * u.deg)
    pole = pole_from_endpoints(c1, c2)
    assert u.allclose(pole.dec, 90 * u.deg)

    c1 = coord.SkyCoord(0 * u.deg, 0 * u.deg)
    c2 = coord.SkyCoord(0 * u.deg, 90 * u.deg)
    pole = pole_from_endpoints(c1, c2)
    assert u.allclose(pole.ra, 270 * u.deg)
    assert u.allclose(pole.dec, 0 * u.deg)

    # Should work even if coord has velocities:
    c1 = coord.SkyCoord(
        0 * u.deg, 0 * u.deg, pm_ra_cosdec=10 * u.mas / u.yr, pm_dec=-0.5 * u.mas / u.yr
    )
    c2 = coord.SkyCoord(
        0 * u.deg,
        90 * u.deg,
        pm_ra_cosdec=10 * u.mas / u.yr,
        pm_dec=-0.5 * u.mas / u.yr,
    )
    pole = pole_from_endpoints(c1, c2)
    assert u.allclose(pole.ra, 270 * u.deg)
    assert u.allclose(pole.dec, 0 * u.deg)


def test_init_pole_from_xyz():
    xnew = coord.UnitSphericalRepresentation(185 * u.deg, 32.5 * u.deg).to_cartesian()
    ynew = coord.UnitSphericalRepresentation(275 * u.deg, 0 * u.deg).to_cartesian()
    znew = xnew.cross(ynew)

    fr1 = GreatCircleICRSFrame.from_xyz(xnew, ynew, znew)
    fr2 = GreatCircleICRSFrame.from_xyz(xnew, ynew)
    fr3 = GreatCircleICRSFrame.from_xyz(xnew, znew=znew)
    fr4 = GreatCircleICRSFrame.from_xyz(ynew=ynew, znew=znew)

    for fr in [fr2, fr3, fr4]:
        assert np.isclose(fr1.pole.ra.degree, fr.pole.ra.degree)
        assert np.isclose(fr1.pole.dec.degree, fr.pole.dec.degree)
        assert np.isclose(fr1.origin.ra.degree, fr.origin.ra.degree)
        assert np.isclose(fr1.origin.dec.degree, fr.origin.dec.degree)

    with pytest.raises(ValueError):
        GreatCircleICRSFrame.from_xyz(xnew)


def test_sph_midpoint():
    c1 = coord.SkyCoord(0 * u.deg, 0 * u.deg)
    c2 = coord.SkyCoord(90 * u.deg, 0 * u.deg)
    midpt = sph_midpoint(c1, c2)
    assert u.allclose(midpt.ra, 45 * u.deg)
    assert u.allclose(midpt.dec, 0 * u.deg)

    c1 = coord.SkyCoord(0 * u.deg, 0 * u.deg)
    c2 = coord.SkyCoord(0 * u.deg, 90 * u.deg)
    midpt = sph_midpoint(c1, c2)
    assert u.allclose(midpt.ra, 0 * u.deg)
    assert u.allclose(midpt.dec, 45 * u.deg)


def test_init_from_R():
    from ..gd1 import GD1Koposov10
    from ..gd1 import R as gd1_R

    N = 128
    rnd = np.random.RandomState(42)

    gd1_gc_frame = GreatCircleICRSFrame.from_R(gd1_R)
    tmp_in = GD1Koposov10(
        phi1=rnd.uniform(0, 360, N) * u.deg, phi2=rnd.uniform(-90, 90, N) * u.deg
    )

    tmp_out = tmp_in.transform_to(gd1_gc_frame)

    assert u.allclose(tmp_in.phi1, tmp_out.phi1)
    assert u.allclose(tmp_in.phi2, tmp_out.phi2)
