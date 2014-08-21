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

class PotentialTestBase(object):

    def test_method_call(self):
        # single
        r = [[1.,0.,0.]]
        pot_val = self.potential.value(r)
        acc_val = self.potential.acceleration(r)

        # multiple
        r = np.random.uniform(size=(nparticles,3))
        pot_val = self.potential.value(r)
        acc_val = self.potential.acceleration(r)

        if self.pypotential is not None:
            py_pot_val = self.potential.value(r)
            py_acc_val = self.potential.acceleration(r)

            assert np.allclose(py_pot_val, pot_val)
            assert np.allclose(py_acc_val, acc_val)
            print("Cython and Python values match")

    def test_orbit_integration(self):
        w0 = self.w0
        t1 = time.time()
        t,w = self.potential.integrate_orbit(w0, dt=1., nsteps=10000)
        print("Cython orbit integration time (10000 steps): {}".format(time.time() - t1))

        if self.pypotential is not None:
            t1 = time.time()
            t,w = self.pypotential.integrate_orbit(w0, dt=1., nsteps=10000)
            print("Python orbit integration time (10000 steps): {}".format(time.time() - t1))

    def test_time_methods(self):
        r = np.random.uniform(size=(nparticles,3))
        for func_name in ["value", "gradient", "acceleration"]:
            t1 = time.time()
            for ii in range(niter):
                x = getattr(self.potential, func_name)(r)
            print("Cython - {}: {:e} sec per call".format(func_name,
                            (time.time()-t1)/float(niter)))

            if self.pypotential is not None:
                t1 = time.time()
                for ii in range(niter):
                    x = getattr(self.pypotential, func_name)(r)
                print("Python - {}: {:e} sec per call".format(func_name,
                                (time.time()-t1)/float(niter)))

    def test_plot_contours(self):

        # test plotting a grid
        grid = np.linspace(-20.,20, 200)

        fig,axes = plt.subplots(1,1)

        t1 = time.time()
        fig,axes = self.potential.plot_contours(grid=(grid,0.,grid),
                                                subplots_kw=dict(figsize=(8,8)))
        print("Cython plot_contours time", time.time() - t1)
        fig.savefig(os.path.join(plot_path, "{}_2d_cy.png"\
                        .format(self.potential.__class__.__name__)))

        if self.pypotential is not None:
            t1 = time.time()
            fig,axes = self.pypotential.plot_contours(grid=(grid,0.,grid),
                                                      subplots_kw=dict(figsize=(8,8)))
            print("Python plot_contours time", time.time() - t1)
            fig.savefig(os.path.join(plot_path, "{}_2d_py.png"\
                         .format(self.pypotential.__class__.__name__)))

class TestHernquist(PotentialTestBase):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def setup(self):
        print()
        from ..builtin import HernquistPotential as PyHernquistPotential

        self.potential = HernquistPotential(usys=self.usys,
                                            m=1.E11, c=0.26)
        self.pypotential = PyHernquistPotential(usys=self.usys,
                                                m=1.E11, c=0.26)
        self.w0 = [1.,0.,0.,0.,0.1,0.1]

class TestMiyamotoNagai(PotentialTestBase):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def setup(self):
        print()
        from ..builtin import MiyamotoNagaiPotential as PyMiyamotoNagaiPotential

        self.potential = MiyamotoNagaiPotential(usys=self.usys,
                                                m=1.E11, a=6.5, b=0.26)
        self.pypotential = PyMiyamotoNagaiPotential(usys=self.usys,
                                                    m=1.E11, a=6.5, b=0.26)
        self.w0 = [8.,0.,0.,0.,0.22,0.1]

class TestLeeSutoNFWPotential(PotentialTestBase):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def setup(self):
        print()
        self.potential = LeeSutoNFWPotential(usys=self.usys,
                                             v_h=0.35, r_h=12.,
                                             a=1.4, b=1., c=0.6)

        self.pypotential = None

        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

