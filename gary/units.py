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
units = dict()
units['length'] = u.kpc
units['speed'] = u.km/u.s
units['time'] = u.Myr
units['angular speed'] = u.mas/u.yr
units['angle'] = u.degree
units['mass'] = u.M_sun
units['dimensionless'] = u.dimensionless_unscaled

# define galactic unit system
galactic = (u.kpc, u.Myr, u.Msun, u.radian)
