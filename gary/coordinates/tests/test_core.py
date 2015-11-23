# coding: utf-8
"""
    Test conversions in core.py
"""

from __future__ import absolute_import, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import pytest
import numpy as np
import tempfile

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename

# This package
from ..core import *

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

    vhel = vgsr_to_vhel(c, vgsr, vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vhel = vgsr_to_vhel(c.transform_to(coord.ICRS), vgsr,
                        vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

    # all together now
    l = coord.Angle(data["lon"] * u.degree)
    b = coord.Angle(data["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vgsr = data["vgsr"] * u.km/u.s
    vhel = vgsr_to_vhel(c, vgsr, vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vhel.value, data['vhelio'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vhel = vgsr_to_vhel(c.transform_to(coord.ICRS), vgsr,
                        vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vhel.value, data['vhelio'], decimal=4)

def test_vgsr_to_vhel_misc():
    # make sure it works with longitude in 0-360 or -180-180
    l1 = coord.Angle(190.*u.deg)
    l2 = coord.Angle(-170.*u.deg)
    b = coord.Angle(30.*u.deg)

    c1 = coord.Galactic(l1, b)
    c2 = coord.Galactic(l2, b)

    vgsr = -110.*u.km/u.s
    vhel1 = vgsr_to_vhel(c1,vgsr)
    vhel2 = vgsr_to_vhel(c2,vgsr)

    np.testing.assert_almost_equal(vhel1.value, vhel2.value, decimal=9)

    # make sure throws error if tuple elements are not Quantities
    with pytest.raises(TypeError):
        vgsr_to_vhel(c1, vgsr.value)

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

    vgsr = vhel_to_vgsr(c, vhel, vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS), vhel,
                        vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

    # all together now
    l = coord.Angle(data["lon"] * u.degree)
    b = coord.Angle(data["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vhel = data["vhelio"] * u.km/u.s
    vgsr = vhel_to_vgsr(c, vhel, vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vgsr.value, data['vgsr'], decimal=4)

    # now check still get right answer passing in ICRS coordinates
    vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS), vhel,
                        vlsr=vlsr, vcirc=vcirc)
    np.testing.assert_almost_equal(vgsr.value, data['vgsr'], decimal=4)

def test_vhel_to_vgsr_misc():
    vhel = 110*u.km/u.s
    c1 = coord.Galactic(15*u.deg, -0.6*u.deg)

    # make sure throws error if tuple elements are not Quantities
    with pytest.raises(TypeError):
        vhel_to_vgsr(c1, vhel.value)

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

    def test_vhel_to_gal_single(self):

        # test a single entry
        row = self.data[0]
        c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, distance=row['dist']*u.pc)
        pm = [row['pml'], row['pmb']]*u.mas/u.yr
        rv = row['rv']*u.km/u.s

        # stupid check
        vxyz_i = vhel_to_gal(c.icrs, pm=pm, rv=rv,
                             vcirc=0*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s)

        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)

        assert vxyz_i.shape == vxyz.shape

        true_UVW = [row['U'],row['V'],row['W']]*u.km/u.s
        found_UVW = vxyz
        np.testing.assert_allclose(true_UVW.value, found_UVW.value, atol=1.)

        # some sanity checks - first, some convenience definitions
        g = coord.Galactic(l=0*u.deg, b=0*u.deg).transform_to(coord.ICRS)
        galcen_frame = coord.Galactocentric(galcen_ra=g.ra,
                                            galcen_dec=g.dec,
                                            z_sun=0*u.kpc)

        # --------------------------------------------------------------------
        # l = 0
        # without LSR and circular velocity
        c = coord.SkyCoord(ra=galcen_frame.galcen_ra, dec=galcen_frame.galcen_dec, distance=2*u.kpc)
        pm = [0., 0]*u.mas/u.yr
        rv = 20*u.km/u.s
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s,
                           galactocentric_frame=galcen_frame)
        np.testing.assert_allclose(vxyz.to(u.km/u.s).value, [20,0,0.], atol=1E-12)

        # with LSR and circular velocity
        c = coord.SkyCoord(ra=galcen_frame.galcen_ra, dec=galcen_frame.galcen_dec, distance=2*u.kpc)
        pm = [0., 0]*u.mas/u.yr
        rv = 20*u.km/u.s
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=galcen_frame)
        np.testing.assert_allclose(vxyz.to(u.km/u.s).value, [0,200,10], atol=1E-12)

        # l = 90
        # with LSR and circular velocity
        c = coord.SkyCoord(l=90*u.deg, b=0*u.deg, distance=2*u.kpc, frame=coord.Galactic)
        pm = [0., 0]*u.mas/u.yr
        rv = 20*u.km/u.s
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=galcen_frame)
        np.testing.assert_allclose(vxyz.to(u.km/u.s).value, [-20,220,10], atol=1E-5)

        # l = 180
        # with LSR and circular velocity
        c = coord.SkyCoord(l=180*u.deg, b=0*u.deg, distance=2*u.kpc, frame=coord.Galactic)
        pm = [0., 0]*u.mas/u.yr
        rv = 20*u.km/u.s
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=galcen_frame)
        np.testing.assert_allclose(vxyz.to(u.km/u.s).value, [-40,200,10], atol=1E-12)

        # l = 270
        # with LSR and circular velocity
        c = coord.SkyCoord(l=270*u.deg, b=0*u.deg, distance=2*u.kpc, frame=coord.Galactic)
        pm = [0., 0]*u.mas/u.yr
        rv = 20*u.km/u.s
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=200*u.km/u.s,
                           vlsr=[-20.,0,10]*u.km/u.s,
                           galactocentric_frame=galcen_frame)
        np.testing.assert_allclose(vxyz.to(u.km/u.s).value, [-20,180,10], atol=1E-5)

    def test_vhel_to_gal_array(self):
        # test all together
        d = self.data
        c = coord.SkyCoord(ra=d['ra']*u.deg, dec=d['dec']*u.deg, distance=d['dist']*u.pc)
        pm = np.vstack((d['pml'], d['pmb']))*u.mas/u.yr
        rv = d['rv']*u.km/u.s

        # stupid check
        vxyz_i = vhel_to_gal(c.icrs, pm=pm, rv=rv,
                             vcirc=0*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s)
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)
        assert vxyz_i.shape == vxyz.shape

        # check values
        true_UVW = np.vstack((d['U'],d['V'],d['W']))*u.km/u.s
        found_UVW = vxyz
        np.testing.assert_allclose(true_UVW.value, found_UVW.value, atol=1.)

    def test_vgal_to_hel_single(self):

        # test a single entry
        row = self.data[0]
        c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, distance=row['dist']*u.pc)
        pm = [row['pml'],row['pmb']]*u.mas/u.yr
        rv = row['rv']*u.km/u.s

        true_pmrv = (pm[0], pm[1], rv)
        vxyz = [row['U'],row['V'],row['W']]*u.km/u.s
        pmrv = vgal_to_hel(c.galactic, vxyz=vxyz,
                           vcirc=0.*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)

        for i in range(3):
            np.testing.assert_allclose(pmrv[i].to(true_pmrv[i].unit).value,
                                       true_pmrv[i].value,
                                       atol=1.)

        # some sanity checks - first, some convenience definitions
        g = coord.Galactic(l=0*u.deg, b=0*u.deg).transform_to(coord.ICRS)
        frargs = dict(galcen_ra=g.ra,
                      galcen_dec=g.dec,
                      z_sun=0*u.kpc,
                      galcen_distance=8*u.kpc)
        galcen_frame = coord.Galactocentric(**frargs)

        # --------------------------------------------------------------------
        # l = 0
        # without LSR and circular velocity
        # c = coord.Galactocentric([6,0,0]*u.kpc,**frargs)
        c = coord.SkyCoord(l=0*u.deg, b=0*u.deg, distance=2*u.kpc, frame=coord.Galactic)
        vxyz = [20.,0,0]*u.km/u.s
        pmv = vgal_to_hel(c.galactic, vxyz,
                          vcirc=0*u.km/u.s,
                          vlsr=[0.,0,0]*u.km/u.s,
                          galactocentric_frame=galcen_frame)
        np.testing.assert_allclose(pmv[0].to(u.mas/u.yr).value, 0., atol=1E-12)
        np.testing.assert_allclose(pmv[1].to(u.mas/u.yr).value, 0., atol=1E-12)
        np.testing.assert_allclose(pmv[2].to(u.km/u.s).value, 20., atol=1E-12)

        # with LSR and circular velocity
        c = coord.SkyCoord(l=0*u.deg, b=0*u.deg, distance=2*u.kpc, frame=coord.Galactic)
        vxyz = [20.,0,0]*u.km/u.s
        pmv = vgal_to_hel(c.galactic, vxyz,
                          vcirc=-200*u.km/u.s,
                          vlsr=[0.,0,10]*u.km/u.s,
                          galactocentric_frame=galcen_frame)

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            np.testing.assert_allclose(pmv[0].to(u.mas/u.yr).value,
                                       ((200.*u.km/u.s)/(2*u.kpc)).to(u.mas/u.yr).value,
                                       atol=1E-12)
            np.testing.assert_allclose(pmv[1].to(u.mas/u.yr).value,
                                       ((-10.*u.km/u.s)/(2*u.kpc)).to(u.mas/u.yr).value,
                                       atol=1E-4)
        np.testing.assert_allclose(pmv[2].to(u.km/u.s).value, 20., atol=1E-12)

    def test_vgal_to_hel_array(self):
        # test all together
        d = self.data
        c = coord.SkyCoord(ra=d['ra']*u.deg, dec=d['dec']*u.deg, distance=d['dist']*u.pc)
        pm = np.vstack([d['pml'],d['pmb']])*u.mas/u.yr
        rv = d['rv']*u.km/u.s

        true_pmrv = (pm[0], pm[1], rv)
        vxyz = np.vstack((d['U'],d['V'],d['W']))*u.km/u.s
        pmrv = vgal_to_hel(c.galactic, vxyz=vxyz,
                           vcirc=0.*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)

        for i in range(3):
            np.testing.assert_allclose(pmrv[i].to(true_pmrv[i].unit).value,
                                       true_pmrv[i].value,
                                       atol=1.)

    def test_roundtrip_icrs(self):
        np.random.seed(42)
        n = 100

        # yeahhhh, i know this isn't uniform on the sphere - shut up
        c = coord.SkyCoord(ra=np.random.uniform(0,360,n)*u.degree,
                           dec=np.random.uniform(-90,90,n)*u.degree,
                           distance=np.random.uniform(0.1,10.,n)*u.kpc)

        pm = np.random.uniform(-20,20,size=(2,n)) * u.mas/u.yr
        vr = np.random.normal(0., 75., size=n)*u.km/u.s
        mua,mud = pm  # initial

        # first to galactocentric
        vxyz = vhel_to_gal(c.icrs, pm=pm, rv=vr)

        # then back again, wooo
        pmv = vgal_to_hel(c.icrs, vxyz=vxyz)

        mua2,mud2 = pmv[:2]
        vr2 = pmv[2]

        np.testing.assert_allclose(mua.to(u.mas/u.yr).value, mua2.to(u.mas/u.yr).value, atol=1e-12)
        np.testing.assert_allclose(mud.to(u.mas/u.yr).value, mud2.to(u.mas/u.yr).value, atol=1e-12)
        np.testing.assert_allclose(vr.to(u.km/u.s).value, vr2.to(u.km/u.s).value, atol=1e-12)

    def test_roundtrip_gal(self):
        np.random.seed(42)
        n = 100

        # yeahhhh, i know this isn't uniform on the sphere - shut up
        c = coord.SkyCoord(ra=np.random.uniform(0,360,n)*u.degree,
                           dec=np.random.uniform(-90,90,n)*u.degree,
                           distance=np.random.uniform(0.1,10.,n)*u.kpc)

        pm = np.random.uniform(-20,20,size=(2,n)) * u.mas/u.yr
        vr = np.random.normal(0., 75., size=n)*u.km/u.s
        mul,mub = pm  # initial

        # first to galactocentric
        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=vr)

        # then back again, wooo
        pmv = vgal_to_hel(c.galactic, vxyz=vxyz)

        mul2,mub2 = pmv[:2]
        vr2 = pmv[2]

        np.testing.assert_allclose(mul.to(u.mas/u.yr).value, mul2.to(u.mas/u.yr).value, rtol=1E-5, atol=1e-12)
        np.testing.assert_allclose(mub.to(u.mas/u.yr).value, mub2.to(u.mas/u.yr).value, rtol=1E-5, atol=1e-12)
        np.testing.assert_allclose(vr.to(u.km/u.s).value, vr2.to(u.km/u.s).value, rtol=1E-5, atol=1e-12)
