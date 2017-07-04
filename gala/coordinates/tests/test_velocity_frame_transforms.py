# coding: utf-8
"""
    Test conversions in core.py
"""

from __future__ import absolute_import, division, print_function


# Standard library
import tempfile

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import quantity_allclose
import numpy as np

# This package
from ..velocity_frame_transforms import (vgal_to_hel, vhel_to_gal,
                                         vgsr_to_vhel, vhel_to_vgsr)

def test_vgsr_to_vhel():
    filename = get_pkg_data_filename('idl_vgsr_vhel.txt')
    data = np.genfromtxt(filename, names=True, skip_header=2)

    # one row
    row = data[0]
    l = coord.Angle(row["lon"] * u.degree)
    b = coord.Angle(row["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vgsr = row["vgsr"] * u.km/u.s
    vlsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s # this is right
    vcirc = row["vcirc"]*u.km/u.s

    vsun = vlsr + [0,1,0]*vcirc
    vhel = vgsr_to_vhel(c, vgsr, vsun=vsun)
    return
    np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vhel = vgsr_to_vhel(c.transform_to(coord.ICRS), vgsr, vsun=vsun)
    np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

    # all together now
    l = coord.Angle(data["lon"] * u.degree)
    b = coord.Angle(data["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vgsr = data["vgsr"] * u.km/u.s
    vhel = vgsr_to_vhel(c, vgsr, vsun=vsun)
    np.testing.assert_almost_equal(vhel.value, data['vhelio'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vhel = vgsr_to_vhel(c.transform_to(coord.ICRS), vgsr, vsun=vsun)
    np.testing.assert_almost_equal(vhel.value, data['vhelio'], decimal=4)

def test_vgsr_to_vhel_misc():
    # make sure it works with longitude in 0-360 or -180-180
    l1 = coord.Angle(190.*u.deg)
    l2 = coord.Angle(-170.*u.deg)
    b = coord.Angle(30.*u.deg)

    c1 = coord.Galactic(l1, b)
    c2 = coord.Galactic(l2, b)

    vgsr = -110.*u.km/u.s
    vhel1 = vgsr_to_vhel(c1, vgsr)
    vhel2 = vgsr_to_vhel(c2, vgsr)

    np.testing.assert_almost_equal(vhel1.value, vhel2.value, decimal=9)

def test_vhel_to_vgsr():
    filename = get_pkg_data_filename('idl_vgsr_vhel.txt')
    data = np.genfromtxt(filename, names=True, skip_header=2)

    # one row
    row = data[0]
    l = coord.Angle(row["lon"] * u.degree)
    b = coord.Angle(row["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vhel = row["vhelio"] * u.km/u.s
    vlsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s # this is right
    vcirc = row["vcirc"]*u.km/u.s

    vsun = vlsr + [0,1,0]*vcirc
    vgsr = vhel_to_vgsr(c, vhel, vsun=vsun)
    np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS), vhel, vsun=vsun)
    np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

    # all together now
    l = coord.Angle(data["lon"] * u.degree)
    b = coord.Angle(data["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vhel = data["vhelio"] * u.km/u.s
    vgsr = vhel_to_vgsr(c, vhel, vsun=vsun)
    np.testing.assert_almost_equal(vgsr.value, data['vgsr'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS), vhel, vsun=vsun)
    np.testing.assert_almost_equal(vgsr.value, data['vgsr'], decimal=4)

_txt = """# from: XHIP catalog
# ra dec HIPID l b dist pml pmb rv U V W
0.022010 20.036114      7 106.82021040 -41.22316218   57.56  -253.69  -138.84    8.30   71.7    2.1  -34.0
2.208349 40.494550    714 114.23363142 -21.65650026  249.00     5.57    -9.00  -11.78    0.1  -16.3   -5.5
3.126297 14.563522    999 108.98177530 -47.25067692   40.94   296.66  -141.05  -15.30  -44.5  -47.6   -7.3
"""
class TestVHelGalConvert(object):

    def setup(self):
        with tempfile.NamedTemporaryFile(mode='w+b') as temp:
            temp.write(_txt.encode('utf-8'))
            temp.flush()
            temp.seek(0)
            self.data = np.genfromtxt(temp, names=True, skip_header=1)

        # This should make the transformations more compatible
        g = coord.Galactic(l=0*u.deg, b=0*u.deg).transform_to(coord.ICRS)
        self.galcen_frame = coord.Galactocentric(galcen_coord=g,
                                                 z_sun=0*u.kpc)

    def test_vhel_to_gal_single(self):

        for row in self.data: # test one entry at a time
            c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg,
                               distance=row['dist']*u.pc)
            icrs = c.icrs
            gal = c.galactic
            pm = [row['pml']*u.mas/u.yr, row['pmb']*u.mas/u.yr,]
            rv = row['rv']*u.km/u.s

            # stupid check
            vxyz_i = vhel_to_gal(icrs, pm=pm, rv=rv,
                                 vcirc=0*u.km/u.s,
                                 vlsr=[0.,0,0]*u.km/u.s)

            vxyz = vhel_to_gal(gal, pm=pm, rv=rv,
                               vcirc=0*u.km/u.s,
                               vlsr=[0.,0,0]*u.km/u.s)

            assert vxyz_i.shape == vxyz.shape

            true_UVW = np.array([row['U'], row['V'], row['W']])
            UVW = vxyz.to(u.km/u.s).value

            # catalog values are rounded
            assert np.allclose(UVW, true_UVW, rtol=1E-2, atol=0.1)

        # --------------------------------------------------------------------
        # l = 0
        # without LSR and circular velocity
        c = coord.SkyCoord(ra=self.galcen_frame.galcen_ra,
                           dec=self.galcen_frame.galcen_dec,
                           distance=2*u.kpc)
        pm = [0,0]*u.mas/u.yr
        rv = 20*u.km/u.s
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)
        assert np.allclose(vxyz.to(u.km/u.s).value, [20,0,0.],
                           atol=1E-12)

        # with LSR and circular velocity
        c = coord.SkyCoord(ra=self.galcen_frame.galcen_ra,
                           dec=self.galcen_frame.galcen_dec,
                           distance=2*u.kpc)
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)
        assert np.allclose(vxyz.to(u.km/u.s).value, [0,200,10],
                           atol=1E-12)

        # l = 90
        # with LSR and circular velocity
        c = coord.SkyCoord(l=90*u.deg, b=0*u.deg,
                           distance=2*u.kpc, frame=coord.Galactic)
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)
        assert np.allclose(vxyz.to(u.km/u.s).value, [-20,220,10],
                           atol=1E-5)

        # l = 180
        # with LSR and circular velocity
        c = coord.SkyCoord(l=180*u.deg, b=0*u.deg,
                           distance=2*u.kpc, frame=coord.Galactic)
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)
        assert np.allclose(vxyz.to(u.km/u.s).value, [-40,200,10],
                           atol=1E-12)

        # l = 270
        # with LSR and circular velocity
        c = coord.SkyCoord(l=270*u.deg, b=0*u.deg,
                           distance=2*u.kpc, frame=coord.Galactic)
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)
        assert np.allclose(vxyz.to(u.km/u.s).value, [-20,180,10],
                           atol=1E-5)

    def test_vhel_to_gal_array(self):
        # test all together
        d = self.data
        c = coord.SkyCoord(ra=d['ra']*u.deg, dec=d['dec']*u.deg,
                           distance=d['dist']*u.pc)
        icrs = c.icrs
        gal = c.galactic
        pm = [d['pml'], d['pmb']]*u.mas/u.yr
        rv = d['rv']*u.km/u.s

        # stupid check
        vxyz_i = vhel_to_gal(icrs, pm=pm, rv=rv,
                             vcirc=0*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s)
        vxyz = vhel_to_gal(gal, pm=pm, rv=rv,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)
        assert vxyz_i.shape == vxyz.shape

        # check values
        true_UVW = np.array([d['U'], d['V'], d['W']])
        UVW = vxyz.to(u.km/u.s).value

        # catalog values are rounded
        assert np.allclose(UVW, true_UVW, rtol=1E-2, atol=0.1)

    def test_vgal_to_hel_single(self):

        for row in self.data: # test one entry at a time
            c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg,
                               distance=row['dist']*u.pc)
            gal = c.galactic
            vxyz = [row['U'], row['V'], row['W']] * u.km/u.s

            vhel = vgal_to_hel(gal, vxyz,
                               vcirc=0.*u.km/u.s,
                               vlsr=[0.,0,0]*u.km/u.s,
                               galactocentric_frame=self.galcen_frame)

            # tolerance set by the catalog rounded numbers
            assert quantity_allclose(vhel[0], row['pml'] * u.mas/u.yr, rtol=1E-2)
            assert quantity_allclose(vhel[1], row['pmb'] * u.mas/u.yr, rtol=1E-2)
            assert quantity_allclose(vhel[2], row['rv'] * u.km/u.s, rtol=1E-2)

        # --------------------------------------------------------------------
        # l = 0
        # without LSR and circular velocity
        c = coord.SkyCoord(l=0*u.deg, b=0*u.deg, distance=2*u.kpc,
                           frame=coord.Galactic)
        vxyz = [20., 0, 0]*u.km/u.s
        vhel = vgal_to_hel(c.galactic, vxyz,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)

        assert np.allclose(vhel[0].value, 0., atol=1E-12)
        assert np.allclose(vhel[1].value, 0., atol=1E-12)
        assert np.allclose(vhel[2].to(u.km/u.s).value, 20., atol=1E-12)

        vxyz = [20., 0, 50]*u.km/u.s
        vhel = vgal_to_hel(c.galactic, vxyz,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)

        assert np.allclose(vhel[0].value, 0., atol=2E-5) # TODO: astropy precision issues
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            assert quantity_allclose(vhel[1], 50*u.km/u.s / (2*u.kpc),
                                     atol=1E-10*u.mas/u.yr)
        assert quantity_allclose(vhel[2].to(u.km/u.s), vxyz[0],
                                 atol=1E-10*u.km/u.s)

        # with LSR and circular velocity
        vxyz = [20., 0, 50]*u.km/u.s
        vhel = vgal_to_hel(c.galactic, vxyz,
                           vcirc=-200*u.km/u.s,
                           vlsr=[0., 0, 10]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            assert quantity_allclose(vhel[0],
                                     (200.*u.km/u.s) / (2*u.kpc),
                                     atol=1E-10*u.mas/u.yr)
            assert quantity_allclose(vhel[1],
                                     (40.*u.km/u.s) / (2*u.kpc),
                                     atol=1E-6*u.mas/u.yr)

        assert quantity_allclose(vhel[2], 20.*u.km/u.s,
                                 atol=1E-10*u.km/u.s)

    def test_vgal_to_hel_array(self):
        # test all together
        d = self.data
        c = coord.SkyCoord(ra=d['ra']*u.deg, dec=d['dec']*u.deg,
                           distance=d['dist']*u.pc)

        pm = np.vstack([d['pml'],d['pmb']])*u.mas/u.yr
        rv = d['rv']*u.km/u.s

        vxyz = np.vstack((d['U'], d['V'], d['W']))*u.km/u.s
        vhel = vgal_to_hel(c.galactic, vxyz,
                           vcirc=0.*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s,
                           galactocentric_frame=self.galcen_frame)

        # tolerance set by the catalog rounded numbers
        assert quantity_allclose(vhel[0], pm[0], rtol=1E-2)
        assert quantity_allclose(vhel[1], pm[1], rtol=1E-2)
        assert quantity_allclose(vhel[2], rv, rtol=5E-3)

    def test_roundtrip_icrs(self):
        np.random.seed(42)
        n = 100

        # yeahhhh, i know this isn't uniform on the sphere - shut up
        c = coord.SkyCoord(ra=np.random.uniform(0,360,n)*u.degree,
                           dec=np.random.uniform(-90,90,n)*u.degree,
                           distance=np.random.uniform(0.1,10.,n)*u.kpc)

        pm = np.random.uniform(-20,20,size=(2,n)) * u.mas/u.yr
        vr = np.random.normal(0., 75., size=n)*u.km/u.s

        # first to galactocentric
        vxyz = vhel_to_gal(c.icrs, pm=pm, rv=vr)

        # then back again, wooo
        vhel2 = vgal_to_hel(c.icrs, vxyz)

        assert quantity_allclose(vhel2[0], pm[0], rtol=1e-12)
        assert quantity_allclose(vhel2[1], pm[1], rtol=1e-12)
        assert quantity_allclose(vhel2[2], vr, rtol=1e-12)

    def test_roundtrip_gal(self):
        np.random.seed(42)
        n = 100

        # yeahhhh, i know this isn't uniform on the sphere - shut up
        c = coord.SkyCoord(ra=np.random.uniform(0,360,n)*u.degree,
                           dec=np.random.uniform(-90,90,n)*u.degree,
                           distance=np.random.uniform(0.1,10.,n)*u.kpc)

        pm = np.random.uniform(-20,20,size=(2,n)) * u.mas/u.yr
        vr = np.random.normal(0., 75., size=n)*u.km/u.s

        # first to galactocentric
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=vr)

        # then back again, wooo
        vhel2 = vgal_to_hel(c.galactic, vxyz)

        # TODO: why such bad roundtripping???
        assert quantity_allclose(vhel2[0], pm[0], rtol=1e-12)
        assert quantity_allclose(vhel2[1], pm[1], rtol=1e-12)
        assert quantity_allclose(vhel2[2], vr, rtol=1e-12)
