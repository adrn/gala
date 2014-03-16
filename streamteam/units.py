# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import astropy.units as u

# Create logger
logger = logging.getLogger(__name__)

# default unit system
usys = dict()
usys['length'] = u.kpc
usys['speed'] = u.km/u.s
usys['time'] = u.Myr
usys['angular speed'] = u.mas/u.yr
usys['angle'] = u.degree
usys['mass'] = u.M_sun
usys['dimensionless'] = u.dimensionless_unscaled