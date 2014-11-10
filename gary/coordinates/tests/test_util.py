# coding: utf-8

""" Test utils. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest
from astropy import log as logger
import astropy.units as u

# Project
from ..util import cartesian_to_spherical, spherical_to_cartesian

def test_roundtrip():
    n = 1000

    # cartesian -> spherical -> cartesian
    x = np.random.uniform(-100,100,size=(n,3))
    v = np.random.uniform(-1,1,size=(n,3))

    sph = cartesian_to_spherical(x, v)
    xp,vp = spherical_to_cartesian(*sph.T)

    assert np.allclose(x, xp)
    assert np.allclose(v, vp)

    # spherical -> cartesian -> spherical
    r = np.random.uniform(0.1,100,size=n)
    phi = np.random.uniform(0.,2*np.pi,size=n)
    theta = np.arccos(2*np.random.uniform(0.,1,size=n) - 1)
    vr,vphi,vtheta = np.random.uniform(-1,1,size=(3,n))

    x,v = spherical_to_cartesian(r,phi,theta,vr,vphi,vtheta)
    sph = cartesian_to_spherical(x, v)

    assert np.allclose(r, sph[:,0])
    assert np.allclose(phi, sph[:,1])
    assert np.allclose(theta, sph[:,2])
    assert np.allclose(vr, sph[:,3])
    assert np.allclose(vphi, sph[:,4])
    assert np.allclose(vtheta, sph[:,5])

