# coding: utf-8
"""
    Test the RR Lyrae helper functions.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..core import *
from ..rrlyrae import *

def test_gaia_rv_error():
    d = np.linspace(1.,50.,100)*u.kpc
    rv_errs = gaia_rv_error(d)

def test_gaia_pm_error():
    d = np.linspace(1.,50.,100)*u.kpc
    pm_errs = gaia_pm_error(d)

    vtan_errs = pm_errs.to(u.rad/u.yr).value/u.yr*d
    vtan_errs = vtan_errs.to(u.km/u.s)


