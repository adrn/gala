# Third-party
import astropy.coordinates as coord
import astropy.units as u

# This package
from ..reflex import reflex_correct

def test_reflex():
    c = coord.SkyCoord(ra=162*u.deg,
                       dec=-17*u.deg,
                       distance=172*u.pc,
                       pm_ra_cosdec=-11*u.mas/u.yr,
                       pm_dec=4*u.mas/u.yr,
                       radial_velocity=110*u.km/u.s)
    c2 = reflex_correct(c)
    c3 = reflex_correct(c, coord.Galactocentric(z_sun=0*u.pc))
