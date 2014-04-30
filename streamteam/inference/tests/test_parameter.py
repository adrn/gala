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
        p = ModelParameter("m", value=np.nan, truth=1.5,
                           prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm' truth=1.5>"
        assert str(p) == "m"

        p = ModelParameter("m", value=np.nan,
                           prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"