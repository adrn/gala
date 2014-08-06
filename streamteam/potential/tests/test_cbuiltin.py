# coding: utf-8
"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.utils.console import color_print
from astropy.constants import G
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

from ..cbuiltin import *

# HACK: bad solution is to do this:
# python setup.py build_ext --inplace

#top_path = "/tmp/streamteam"
top_path = "plots/"
plot_path = os.path.join(top_path, "tests/potential/cpotential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

usys = [u.kpc,u.Myr,u.Msun,u.radian]
G = G.decompose(usys)

print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")

def test_simple():
    return
