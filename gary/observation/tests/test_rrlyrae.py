# coding: utf-8
"""
    Test the RR Lyrae helper functions.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
import pytest
try:
    import pygaia
    HAS_PYGAIA = True
except ImportError:
    HAS_PYGAIA = False

# This project
from ..core import *
from ..rrlyrae import *

@pytest.mark.skipif(not HAS_PYGAIA, reason="pygaia not installed")
def test_gaia_rv_error():
    d = np.linspace(1.,50.,100)*u.kpc
    rv_errs = gaia_radial_velocity_error(d)

@pytest.mark.skipif(not HAS_PYGAIA, reason="pygaia not installed")
def test_gaia_pm_error():
    d = np.linspace(1.,50.,100)*u.kpc
    pm_errs = gaia_proper_motion_error(d)

    vtan_errs = pm_errs.to(u.rad/u.yr).value/u.yr*d
    vtan_errs = vtan_errs.to(u.km/u.s)


