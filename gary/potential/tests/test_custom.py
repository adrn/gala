# coding: utf-8
"""
    Test the LM10Potential
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..custom import *
from .test_cbuiltin import PotentialTestBase

top_path = "/tmp/gary"
plot_path = os.path.join(top_path, "tests/potential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from astropy.utils.console import color_print
print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")

class TestPW14Potential(PotentialTestBase):
    units = (u.kpc, u.M_sun, u.Myr, u.radian)

    def setup(self):
        print("\n\n")
        print("="*50)
        print(self.__class__.__name__)

        self.potential = PW14Potential()
        self.w0 = [8.,0.,0.,0.,0.22,0.1]

class TestLM10Potential(PotentialTestBase):
    units = (u.kpc, u.M_sun, u.Myr, u.radian)

    def setup(self):
        print("\n\n")
        print("="*50)
        print(self.__class__.__name__)

        self.potential = LM10Potential()
        self.w0 = [8.,0.,0.,0.,0.22,0.1]

class TestTriaxialMWPotential(PotentialTestBase):
    units = (u.kpc, u.M_sun, u.Myr, u.radian)

    def setup(self):
        print("\n\n")
        print("="*50)
        print(self.__class__.__name__)

        self.potential = TriaxialMWPotential()
        self.w0 = [8.,0.,0.,0.,0.22,0.1]