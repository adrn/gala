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

    pos = np.random.uniform(-10,10,size=(3,n)) * u.kpc
    vel = np.random.uniform(-100,100,size=(3,n)) * u.km/u.s
    pos_repr = coord.CartesianRepresentation(pos)

    # dimensionless
    vsph1 = cartesian_to_spherical(pos.value * u.dimensionless_unscaled,
                                   vel.value * u.dimensionless_unscaled)
    assert vsph1.unit == u.dimensionless_unscaled

    # astropy coordinates
    cpos = coord.SkyCoord(pos_repr)
    vsph2 = cartesian_to_spherical(cpos, vel)
    assert vsph2.unit == u.km/u.s

    # astropy representation
    vsph3 = cartesian_to_spherical(pos_repr, vel)
    assert vsph3.unit == u.km/u.s

    np.testing.assert_allclose(vsph1, vsph2)
    np.testing.assert_allclose(vsph1, vsph3)
