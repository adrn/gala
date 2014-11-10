# coding: utf-8
"""
    Test the Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import *
from ..builtin import *

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

class TestHarmonicOscillator(object):

    def test_eval(self):
        potential = HarmonicOscillatorPotential(omega=1.)

        # 1D oscillator, a single position
        r = 1.
        pot_val = potential.value(r)
        assert np.allclose(pot_val, 0.5, atol=5)

        acc_val = potential.acceleration(r)
        assert np.allclose(acc_val, -1., atol=5)

        # 2D oscillator, single position
        r = [1.,0.75]
        potential = HarmonicOscillatorPotential(omega=[1.,2.])
        pot_val = potential.value(r)
        assert np.allclose(pot_val, 1.625)

        # 2D oscillator, multiple positions
        r = [[1.,0.75],[2.,1.4],[1.5,0.1]]
        pot_val = potential.value(r)
        assert np.allclose(pot_val, [1.625,5.92,1.145])
        acc_val = potential.acceleration(r)
        assert acc_val.shape == (3,2)

    def test_plot(self):
        potential = HarmonicOscillatorPotential(omega=[1.,2.])
        grid = np.linspace(-5.,5)

        fig = potential.plot_contours(grid=(grid,0.))
        fig.savefig(os.path.join(plot_path, "harmonic_osc_1d.png"))

        fig = potential.plot_contours(grid=(grid,grid))
        fig.savefig(os.path.join(plot_path, "harmonic_osc_2d.png"))

class TestPointMass(object):

    def test_pointmass_creation(self):
        potential = PointMassPotential(m=1.,x0=[0.,0.,0.])

        # no mass provided
        with pytest.raises(TypeError):
            potential = PointMassPotential(x0=[0.,0.,0.])

    def test_pointmass_eval(self):
        potential = PointMassPotential(m=1., x0=[0.,0.,0.],
                                       units=[u.M_sun, u.yr, u.au])

        # Test with a single position
        r = [1.,0.,0.]
        pot_val = potential.value(r)
        assert np.allclose(pot_val, -39.487906, atol=5)

        acc_val = potential.acceleration(r)
        assert np.allclose(acc_val, [-39.487906,0.,0.], atol=5)

    def test_pointmass_plot(self):
        potential = PointMassPotential(m=1., x0=[0.,0.,0.],
                                       units=[u.M_sun, u.yr, u.au])
        grid = np.linspace(-5.,5)

        fig = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "point_mass_1d.png"))

        fig = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "point_mass_2d.png"))

class TestIsochrone(object):
    units = (u.kpc, u.M_sun, u.Myr, u.radian)

    def test_create_plot(self):

        potential = IsochronePotential(units=self.units,
                                       m=1.E11, b=5.)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        axes = None
        grid = np.linspace(-20.,20, 50)
        for slc in np.linspace(-20.,0.,10):
            if axes is None:
                fig = potential.plot_contours(grid=(grid,slc,0.), marker=None)
                axes = fig.axes
            else:
                potential.plot_contours(grid=(grid,slc,0.), ax=axes, marker=None)
        fig.savefig(os.path.join(plot_path, "isochrone_1d.png"))

class TestComposite(object):
    units = (u.au, u.M_sun, u.yr)

    def test_composite_create(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(units=self.units,
                                              m=1., x0=[0.,0.,0.])

        with pytest.raises(TypeError):
            potential["two"] = "derp"

    def test_plot_composite(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(units=self.units,
                                              m=1., x0=[1.,1.,0.])
        potential["two"] = PointMassPotential(units=self.units,
                                              m=1., x0=[-1.,-1.,0.])

        # Where forces cancel
        np.testing.assert_array_almost_equal(
                        potential.acceleration([0.,0.,0.]),
                        [0.,0.,0.], decimal=5)

        grid = np.linspace(-5.,5)
        fig = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "two_equal_point_masses_1d.png"))

        fig = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "two_equal_point_masses_2d.png"))

    def test_plot_composite_mass_ratio(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(units=self.units,
                                              m=1., x0=[1.,1.,0.])
        potential["two"] = PointMassPotential(units=self.units,
                                              m=5., x0=[-1.,-1.,0.])

        grid = np.linspace(-5.,5)
        fig = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "two_different_point_masses_1d.png"))

        fig = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "two_different_point_masses_2d.png"))

    def test_many_point_masses(self, N=20):
        potential = CompositePotential()

        for ii in range(N):
            r0 = np.random.uniform(-1., 1., size=3)
            r0[2] = 0. # x-y plane
            potential[str(ii)] = PointMassPotential(units=self.units,
                                                    m=np.exp(np.random.uniform(np.log(0.1),0.)),
                                                    x0=r0)

        grid = np.linspace(-1.,1,50)
        fig = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "many_point_mass.png"))
