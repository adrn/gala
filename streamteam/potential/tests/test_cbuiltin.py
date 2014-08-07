# coding: utf-8
"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import time
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

niter = 1000
nparticles = 1000

class TestMiyamotoNagai(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def setup(self):
        print()

    def test_create_plot(self):
        from ..builtin import MiyamotoNagaiPotential as PyMiyamotoNagaiPotential

        potential = MiyamotoNagaiPotential(usys=self.usys,
                                           m=1.E11,
                                           a=6.5,
                                           b=0.26)
        pypotential = PyMiyamotoNagaiPotential(usys=self.usys,
                                               m=1.E11,
                                               a=6.5,
                                               b=0.26)

        # single
        r = [[1.,0.,0.]]
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        # multiple
        r = np.random.uniform(size=(nparticles,3))

        for func_name in ["value", "gradient", "acceleration"]:
            t1 = time.time()
            for ii in range(niter):
                x = getattr(potential, func_name)(r)
            print("Cython - {}: {:e} sec per call".format(func_name,
                            (time.time()-t1)/float(niter)))

            t1 = time.time()
            for ii in range(niter):
                x = getattr(pypotential, func_name)(r)
            print("Python - {}: {:e} sec per call".format(func_name,
                            (time.time()-t1)/float(niter)))
        return
        # acc_val = potential.acceleration(r)

        grid = np.linspace(-20.,20, 200)

        t1 = time.time()
        fig,axes = potential.plot_contours(grid=(grid,0.,grid))
        print(time.time() - t1)

        t1 = time.time()
        fig,axes = pypotential.plot_contours(grid=(grid,0.,grid))
        print(time.time() - t1)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_2d.png"))

class TestLeeSutoNFWPotential(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def setup(self):
        print()

    def test_create_plot(self):
        potential = LeeSutoNFWPotential(usys=self.usys,
                                        v_h=0.125, r_h=12.,
                                        a=1.4, b=1., c=0.6)

        # single
        r = [[1.,0.,0.]]
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        # multiple
        r = np.random.uniform(size=(nparticles,3))
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        for func_name in ["value", "gradient", "acceleration"]:
            t1 = time.time()
            for ii in range(niter):
                x = getattr(potential, func_name)(r)
            print("Cython - {}: {:e} sec per call".format(func_name,
                            (time.time()-t1)/float(niter)))

        # acc_val = potential.acceleration(r)

        grid = np.linspace(-50, 50, 200)

        t1 = time.time()
        fig,axes = potential.plot_contours(grid=(grid,0.,grid))
        print(time.time() - t1)

        fig.savefig(os.path.join(plot_path, "lee_suto_nfw_2d.png"))
