# coding: utf-8
"""
    Test the core Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third party
import pytest
import numpy as np
from astropy.constants import G
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import matplotlib.pyplot as pl
from matplotlib import cm

# This package
from ..core import PotentialBase, CompositePotential
from ...units import UnitSystem

units = [u.kpc,u.Myr,u.Msun,u.radian]
G = G.decompose(units)

def test_new_simple():

    class MyPotential(PotentialBase):
        def __init__(self, units=None):
            super(MyPotential, self).__init__(parameters={},
                                              units=units)

        def _value(self, r, t=0.):
            return -1/r

        def _gradient(self, r, t=0.):
            return r**-2

    p = MyPotential()
    assert p(0.5) == -2.
    assert p.value(0.5) == -2.
    assert p.acceleration(0.5) == -4.

    p(np.arange(0.5, 11.5, 0.5))
    p.value(np.arange(0.5, 11.5, 0.5))
    p.acceleration(np.arange(0.5, 11.5, 0.5))

# ----------------------------------------------------------------------------

usys = UnitSystem(u.au, u.yr, u.Msun, u.radian)
class MyPotential(PotentialBase):
    def __init__(self, m, x0, units=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['x0'] = np.array(x0)
        super(MyPotential, self).__init__(parameters=parameters,
                                          units=units)

    def _value(self, x, t):
        m = self.parameters['m']
        x0 = self.parameters['x0']
        r = np.sqrt(np.sum((x-x0[:,None])**2, axis=0))
        return -m/r

    def _gradient(self, x, t):
        m = self.parameters['m']
        x0 = self.parameters['x0']
        r = np.sqrt(np.sum((x-x0[:,None])**2, axis=0))
        return m*(x-x0[:,None])/r**3

def test_repr():
    p = MyPotential(m=1.E10*u.Msun, x0=0., units=usys)
    _repr = p.__repr__()
    assert _repr.startswith("<MyPotential: m=")
    assert "m=1" in _repr
    assert "x0=0" in _repr
    assert _repr.endswith("(AU,yr,solMass,rad)>")
    # assert p.__repr__() == "<MyPotential: m=1.00e+10, x0=0.00e+00 (AU,yr,solMass,rad)>"

def test_plot():
    p = MyPotential(m=1, x0=[1.,3.,0.], units=usys)
    f = p.plot_contours(grid=(np.linspace(-10., 10., 100), 0., 0.),
                        labels=["X"])
    # f.suptitle("slice off from 0., won't have cusp")
    # f.savefig(os.path.join(plot_path, "contour_x.png"))

    f = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                              np.linspace(-10., 10., 100),
                              0.),
                        cmap=cm.Blues)
    # f.savefig(os.path.join(plot_path, "contour_xy.png"))

    f = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                              1.,
                              np.linspace(-10., 10., 100)),
                        cmap=cm.Blues, labels=["X", "Z"])
    # f.savefig(os.path.join(plot_path, "contour_xz.png"))

def test_composite():
    p1 = MyPotential(m=1., x0=[1.,0.,0.], units=usys)
    p2 = MyPotential(m=1., x0=[-1.,0.,0.], units=usys)

    p = CompositePotential(one=p1, two=p2)
    assert quantity_allclose(p.value([0.,0.,0.]), -2*usys['energy']/usys['mass'])
    assert quantity_allclose(p.acceleration([0.,0.,0.]), 0.*usys['acceleration'])

    p1 = MyPotential(m=1., x0=[1.,0.,0.], units=usys)
    p2 = MyPotential(m=1., x0=[-1.,0.,0.], units=[u.kpc, u.yr, u.Msun, u.radian])
    with pytest.raises(ValueError):
        p = CompositePotential(one=p1, two=p2)

    p1 = MyPotential(m=1., x0=[1.,0.,0.], units=usys)
    p2 = MyPotential(m=1., x0=[-1.,0.,0.], units=usys)
    p = CompositePotential(one=p1, two=p2)
    assert u.au in p.units
    assert u.yr in p.units
    assert u.Msun in p.units
