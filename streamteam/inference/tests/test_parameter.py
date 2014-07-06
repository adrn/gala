# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime
import copy

# Third-party
import emcee
import numpy as np
import pytest
import astropy.units as u
from astropy.io.misc import fnpickle
import matplotlib.pyplot as plt

from ..core import *
from ..parameter import *
from ..prior import *

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestModelParameter(object):

    def test_string(self):
        p = ModelParameter("m", truth=1.5, prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"

        p = ModelParameter("m", prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"

    def test_dumb(self):
        p = ModelParameter("test")
        assert p.name == "test"
        assert p.frozen == False

    def test_truthvalue(self):
        p = ModelParameter("test", truth=15.)
        assert p.truth.value == 15.
        assert p.truth.unit == u.dimensionless_unscaled

        p = ModelParameter("test", truth=15.*u.km)
        assert p.truth.value == 15.
        assert p.truth.unit == u.km

        p = ModelParameter("test", truth=np.ones(15)*21.*u.km)
        assert p.truth.shape == (15,)
        assert p.truth.unit == u.km

        p = ModelParameter("test", shape=(15,))
        assert p.truth.shape == (15,)
        assert p.truth.unit == u.dimensionless_unscaled

        # one with units, one without not allowed
        with pytest.raises(ValueError):
            p = ModelParameter("test", truth=11*u.km, shape=(15,))

    def test_prior(self):
        prior = LogUniformPrior(a=10., b=25.)
        p = ModelParameter("test", truth=21., prior=prior)