# coding: utf-8
"""
    Test conversions in core.py
"""

from __future__ import absolute_import, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from ..core import *

this_path = os.path.split(__file__)[0]
data = np.genfromtxt(os.path.join(this_path, "idl_vgsr_vhel.txt"),
                     names=True, skiprows=2)

def test_gsr_to_hel():
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

def test_gsr_to_hel_lon():
    l1 = coord.Angle(190.*u.deg)
    l2 = coord.Angle(-170.*u.deg)
    b = coord.Angle(30.*u.deg)

    c1 = coord.Galactic(l1, b)
    c2 = coord.Galactic(l2, b)

    vgsr = -110.*u.km/u.s

    vhel1 = vgsr_to_vhel(c1,vgsr)
    vhel2 = vgsr_to_vhel(c2,vgsr)

    np.testing.assert_almost_equal(vhel1.value, vhel2.value, decimal=9)

def test_hel_to_gsr():
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