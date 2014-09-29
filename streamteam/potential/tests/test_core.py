# coding: utf-8
"""
    Test the core Potential classes
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

from ..core import Potential, CompositePotential

top_path = "/tmp/streamteam"
plot_path = os.path.join(top_path, "tests/potential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

units = [u.kpc,u.Myr,u.Msun,u.radian]
G = G.decompose(units)

print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")

def test_simple():

    def f(r):
        return -1./r

    def gradient(r):
        return r**-2

    p = Potential(func=f, gradient=gradient)
    assert p(0.5) == -2.
    assert p.value(0.5) == -2.
    assert p.acceleration(0.5) == -4.

    p(np.arange(0.5, 11.5, 0.5))
    p.value(np.arange(0.5, 11.5, 0.5))
    p.acceleration(np.arange(0.5, 11.5, 0.5))

def test_repr():

    def f(r,m):
        return -G*m/r

    def gradient(r,m):
        return G*m/r**2

    p = Potential(func=f, gradient=gradient, parameters=dict(m=1.E10*u.Msun))
    assert p.__repr__() == "<Potential: m=1.00e+10 solMass>"

def test_plot():
    def f(x, m, x0):
        r = np.sqrt(np.sum((x-x0)**2, axis=-1))
        return -m/r

    def gradient(x, m, x0):
        r = np.sqrt(np.sum((x-x0)**2, axis=-1))
        return m*(x-x0)/r**3

    p = Potential(func=f, gradient=gradient,
                  parameters=dict(m=1,
                                  x0=[1.,3.,0.]))
    f,a = p.plot_contours(grid=(np.linspace(-10., 10., 100), 0., 0.),
                          labels=["X"])
    f.suptitle("slice off from 0., won't have cusp")
    f.savefig(os.path.join(plot_path, "contour_x.png"))

    f,a = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                                np.linspace(-10., 10., 100),
                                0.),
                          cmap=cm.Blues)
    f.savefig(os.path.join(plot_path, "contour_xy.png"))

    f,a = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                                1.,
                                np.linspace(-10., 10., 100)),
                          cmap=cm.Blues, labels=["X", "Z"])
    f.savefig(os.path.join(plot_path, "contour_xz.png"))

def test_composite():

    def f(x, m, x0):
        r = np.sqrt(np.sum((x-x0)**2, axis=-1))
        return -m/r

    def gradient(x, m, x0):
        r = np.sqrt(np.sum((x-x0)**2, axis=-1))
        return m*(x-x0)/r**3

    p1 = Potential(func=f, gradient=gradient,
                   parameters=dict(m=1.,
                                   x0=[1.,0.,0.]))

    p2 = Potential(func=f, gradient=gradient,
                   parameters=dict(m=1.,
                                   x0=[-1.,0.,0.]))

    p = CompositePotential(one=p1, two=p2)
    assert np.allclose(p.value([0.,0.,0.]), -2)
    assert np.allclose(p.acceleration([0.,0.,0.]), 0.)

    fig, axes = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                                      np.linspace(-10., 10., 100),
                                      0.))
    fig.savefig(os.path.join(plot_path, "composite_point_mass.png"))
