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

top_path = "/tmp/streamteam"
plot_path = os.path.join(top_path, "tests/potential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from astropy.utils.console import color_print
print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")

class TestPointMass(object):

    def test_pointmass_creation(self):
        potential = PointMassPotential(m=1.*u.M_sun,
                                       x0=[[0.,0.,0.]]*u.au)

        # no mass provided
        with pytest.raises(TypeError):
            potential = PointMassPotential(x0=[[0.,0.,0.]]*u.au)


    def test_pointmass_eval(self):
        potential = PointMassPotential(m=1., x0=[0.,0.,0.],
                                       usys=[u.M_sun, u.yr, u.au])

        # Test with a single position
        r = [1.,0.,0.]
        pot_val = potential.value_at(r)
        assert np.allclose(pot_val, -39.487906, atol=5)

        acc_val = potential.acceleration_at(r)
        assert np.allclose(acc_val, [-39.487906,0.,0.], atol=5)

    def test_pointmass_plot(self):
        potential = PointMassPotential(m=1., x0=[0.,0.,0.],
                                       usys=[u.M_sun, u.yr, u.au])
        grid = np.linspace(-5.,5)

        fig,axes = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "point_mass_1d.png"))

        fig,axes = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "point_mass_2d.png"))

class TestComposite(object):
    usys = (u.au, u.M_sun, u.yr)

    def test_composite_create(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(usys=self.usys,
                                              m=1., x0=[0.,0.,0.])

        with pytest.raises(TypeError):
            potential["two"] = "derp"

    def test_plot_composite(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(usys=self.usys,
                                              m=1., x0=[1.,1.,0.])
        potential["two"] = PointMassPotential(usys=self.usys,
                                              m=1., x0=[-1.,-1.,0.])

        # Where forces cancel
        np.testing.assert_array_almost_equal(
                        potential.acceleration_at([0.,0.,0.]),
                        [0.,0.,0.], decimal=5)

        grid = np.linspace(-5.,5)
        fig,axes = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "two_equal_point_masses_1d.png"))

        fig,axes = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "two_equal_point_masses_2d.png"))

    def test_plot_composite_mass_ratio(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(usys=self.usys,
                                              m=1., x0=[1.,1.,0.])
        potential["two"] = PointMassPotential(usys=self.usys,
                                              m=5., x0=[-1.,-1.,0.])

        grid = np.linspace(-5.,5)
        fig,axes = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "two_different_point_masses_1d.png"))

        fig,axes = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "two_different_point_masses_2d.png"))

    def test_many_point_masses(self, N=20):
        potential = CompositePotential()

        for ii in range(N):
            r0 = np.random.uniform(-1., 1., size=3)
            r0[2] = 0. # x-y plane
            potential[str(ii)] = PointMassPotential(usys=self.usys,
                                                    m=np.exp(np.random.uniform(np.log(0.1),0.)),
                                                    x0=r0)

        grid = np.linspace(-1.,1,50)
        fig,axes = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "many_point_mass.png"))

class TestMiyamotoNagai(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_miyamoto_creation(self):

        potential = MiyamotoNagaiPotential(units=self.usys,
                                           m=1.E11*u.M_sun,
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           r_0=[0.,0.,0.]*u.kpc)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai.png"))

    def test_composite(self):
        potential = CompositePotential(units=self.usys)
        potential["disk"] = MiyamotoNagaiPotential(units=self.usys,
                                           m=1.E11*u.M_sun,
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           r_0=[0.,0.,0.]*u.kpc)
        potential["imbh"] = PointMassPotential(units=self.usys,
                                              m=2E9*u.M_sun,
                                              r_0=[5.,5.,0.]*u.kpc)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_imbh.png"))

class TestHernquist(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = HernquistPotential(units=self.usys,
                                       m=1.E11*u.M_sun,
                                       c=10.*u.kpc)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "hernquist.png"))

class TestLogarithmicPotentialLJ(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = LogarithmicPotentialLJ(units=self.usys,
                                           q1=1.4,
                                           q2=1.,
                                           qz=1.5,
                                           phi=1.69*u.radian,
                                           v_halo=120.*u.km/u.s,
                                           R_halo=12.*u.kpc)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "log_halo_lj.png"))

class TestCompositeGalaxy(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_creation(self):
        potential = CompositePotential(units=self.usys)
        potential["disk"] = MiyamotoNagaiPotential(units=self.usys,
                                           m=1.E11*u.M_sun,
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           r_0=[0.,0.,0.]*u.kpc)

        potential["bulge"] = HernquistPotential(units=self.usys,
                                       m=1.E11*u.M_sun,
                                       c=0.7*u.kpc)

        potential["halo"] = LogarithmicPotentialLJ(units=self.usys,
                                           q1=1.4,
                                           q2=1.,
                                           qz=1.5,
                                           phi=1.69*u.radian,
                                           v_halo=120.*u.km/u.s,
                                           R_halo=12.*u.kpc)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid, ndim=3)
        fig.savefig(os.path.join(plot_path, "composite_galaxy.png"))

class TestIsochrone(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = IsochronePotential(units=self.usys,
                                       m=1.E11*u.M_sun,
                                       b=5.*u.kpc)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "isochrone.png"))

class TestAxisymmetricNFWPotential(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = AxisymmetricNFWPotential(units=self.usys,
                                           log_m=28.,
                                           qz=0.71,
                                           Rs=5.*u.kpc)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "nfw.png"))

class TestAxisymmetricLogarithmicPotential(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = AxisymmetricLogarithmicPotential(units=self.usys,
                                           v_c=10.*u.km/u.s,
                                           qz=0.71)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value_at(r)
        acc_val = potential.acceleration_at(r)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "axisym_log.png"))

