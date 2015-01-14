# coding: utf-8
"""
    Test conversions in core.py
"""

from __future__ import absolute_import, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
import tempfile

import astropy.coordinates as coord
import astropy.units as u

from ..core import *

this_path = os.path.split(__file__)[0]
data = np.genfromtxt(os.path.join(this_path, "idl_vgsr_vhel.txt"),
                     names=True, skiprows=2)

def test_vgsr_to_vhel():
    for row in data:
        l = coord.Angle(row["lon"] * u.degree)
        b = coord.Angle(row["lat"] * u.degree)
        c = coord.Galactic(l, b)
        vgsr = row["vgsr"] * u.km/u.s
        vlsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s

        vhel = vgsr_to_vhel(c, vgsr,
                            vlsr=vlsr,
                            vcirc=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

        # now check still get right answer passing in ICRS coordinates
        vhel = vgsr_to_vhel(c.transform_to(coord.ICRS), vgsr,
                            vlsr=vlsr,
                            vcirc=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

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
    for row in data:
        l = coord.Angle(row["lon"] * u.degree)
        b = coord.Angle(row["lat"] * u.degree)
        c = coord.Galactic(l, b)
        vhel = row["vhelio"] * u.km/u.s
        vlsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s

        vgsr = vhel_to_vgsr(c, vhel,
                            vlsr=vlsr,
                            vcirc=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

        # now check still get right answer passing in ICRS coordinates
        vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS), vhel,
                            vlsr=vlsr,
                            vcirc=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

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
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(_txt)
            temp.flush()
            temp.seek(0)
            self.data = np.genfromtxt(temp, names=True, skiprows=1)

    def test_vhel_to_gal(self):

        # test a single entry
        row = self.data[0]
        c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, distance=row['dist']*u.pc)
        pm = [row['pml'],row['pmb']]*u.mas/u.yr
        rv = row['rv']*u.km/u.s

        vxyz = vhel_to_gal(c.galactic, pm=pm, rv=rv,
                           vcirc=0*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)

        true_UVW = [row['U'],row['V'],row['W']]*u.km/u.s
        found_UVW = vxyz.T[0]

        print(true_UVW - found_UVW)

    def test_vgal_to_hel(self):

        # test a single entry
        row = self.data[0]
        c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, distance=row['dist']*u.pc)
        pm = [row['pml'],row['pmb']]*u.mas/u.yr
        rv = row['rv']*u.km/u.s

        true_pmrv = (pm[0],pm[1],rv)
        vxyz = [row['U'],row['V'],row['W']]*u.km/u.s
        pmrv = vgal_to_hel(c.galactic, vxyz=vxyz,
                           vcirc=0.*u.km/u.s,
                           vlsr=[0.,0,0]*u.km/u.s)

        print(pmrv)
        print(true_pmrv)

    def test_roundtrip(self):
        np.random.seed(42)
        n = 2

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

        print(mua - mua2)
        print(mud - mud2)
        print(vr - vr2)

'''
# def test_vhel_to_gal():

#     # test with single
#     c = coord.SkyCoord(ra=100.68458*u.deg, dec=41.26917*u.deg, distance=1.1*u.kpc)

#     pm = (1.5*u.mas/u.yr, -1.7*u.mas/u.yr)
#     rv = 151.1*u.km/u.s
#     vxyz = vhel_to_gal(c, pm=pm, rv=rv)
#     print(vxyz)

# ------------------------
# TODO: all these mofos are dead
def test_gal_to_hel_call():

    r = np.random.uniform(-10,10,size=(3,1000))*u.kpc
    v = np.random.uniform(-100,100,size=(3,1000))*u.km/u.s

    gal_xyz_to_hel(r)
    gal_xyz_to_hel(r, v)

def test_hel_to_gal():

    # l = 0
    r,v = hel_to_gal_xyz(coord.Galactic(0*u.deg, 0*u.deg, distance=2*u.kpc),
                             pm=(0*u.mas/u.yr, 0*u.mas/u.yr), vr=20*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-6,0,0]*u.kpc)
    np.testing.assert_almost_equal(v, [20,200,0]*u.km/u.s)

    # l = 90
    r,v = hel_to_gal_xyz(coord.Galactic(90*u.deg, 0*u.deg, distance=2*u.kpc),
                             pm=(0*u.mas/u.yr, 0*u.mas/u.yr), vr=20*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-8,2,0]*u.kpc)
    np.testing.assert_almost_equal(v, [0,220,0]*u.km/u.s)

    # l = 180
    r,v = hel_to_gal_xyz(coord.Galactic(180*u.deg, 0*u.deg, distance=2*u.kpc),
                             pm=(0*u.mas/u.yr, 0*u.mas/u.yr), vr=20*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-10,0,0]*u.kpc)
    np.testing.assert_almost_equal(v, [-20,200,0]*u.km/u.s)

    # l = 270
    r,v = hel_to_gal_xyz(coord.Galactic(270*u.deg, 0*u.deg, distance=2*u.kpc),
                             pm=(0*u.mas/u.yr, 0*u.mas/u.yr), vr=20*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-8,-2,0]*u.kpc)
    np.testing.assert_almost_equal(v, [0,180,0]*u.km/u.s)

    print(r,v)

def test_gal_to_hel():

    # l = 0
    r,v = gal_xyz_to_hel([-6,0,0]*u.kpc,
                             [20,200,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r.l, 0*u.deg)
    np.testing.assert_almost_equal(r.b, 0*u.deg)
    np.testing.assert_almost_equal(u.Quantity(r.distance), 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    # l = 90
    r,v = gal_xyz_to_hel([-8,2,0]*u.kpc,
                             [0,220,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r.l, 90*u.deg)
    np.testing.assert_almost_equal(r.b, 0*u.deg)
    np.testing.assert_almost_equal(u.Quantity(r.distance), 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    # l = 180
    r,v = gal_xyz_to_hel([-10,0,0]*u.kpc,
                             [-20,200,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r.l, 180*u.deg)
    np.testing.assert_almost_equal(r.b, 0*u.deg)
    np.testing.assert_almost_equal(u.Quantity(r.distance), 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    # l = 270
    r,v = gal_xyz_to_hel([-8,-2,0]*u.kpc,
                             [0,180,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r.l, 270*u.deg)
    np.testing.assert_almost_equal(r.b, 0*u.deg)
    np.testing.assert_almost_equal(u.Quantity(r.distance), 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    print(r,v)

'''
