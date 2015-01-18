# coding: utf-8

""" Test velocity transformations.  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u

# Project
from ..velocity_transforms import *

logger.setLevel(logging.DEBUG)

def test_cartesian_to_spherical():
    n = 10

    # just arrays
    pos = np.random.uniform(-10,10,size=(3,n))
    vel = np.random.uniform(-0.1,0.1,size=(3,n))
    cartesian_to_spherical(pos, vel)

    # # astropy coordinates
    # pos = coord.SkyCoord(ra=np.random.uniform(0,360,size=n)*u.deg,
    #                      dec=np.random.uniform(-90,90,size=n)*u.deg,
    #                      distance=np.random.uniform(0,30,size=n)*u.kpc)
    # vel = np.random.uniform(-0.1,0.1,size=(3,n))
    # cartesian_to_spherical(pos, vel)

    # # astropy representation
    # xyz = np.random.uniform(-10,10,size=(3,n)) * u.dimensionless_unscaled
    # pos = coord.CartesianRepresentation(xyz)
    # vel = np.random.uniform(-0.1,0.1,size=(3,n))
    # cartesian_to_spherical(pos, vel)
