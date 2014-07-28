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
        pot_val = potential.value(r)
        assert np.allclose(pot_val, -39.487906, atol=5)

        acc_val = potential.acceleration(r)
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
                        potential.acceleration([0.,0.,0.]),
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

class TestIsochrone(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = IsochronePotential(usys=self.usys,
                                       m=1.E11, b=5.)

        r = ([1.,0.,0.]*u.kpc).reshape(1,3)
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        axes = None
        grid = np.linspace(-20.,20, 50)
        for slc in np.linspace(-20.,0.,10):
            if axes is None:
                fig,axes = potential.plot_contours(grid=(grid,slc,0.), marker=None)
            else:
                potential.plot_contours(grid=(grid,slc,0.), ax=axes, marker=None)
        fig.savefig(os.path.join(plot_path, "isochrone_1d.png"))

class TestMiyamotoNagai(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = MiyamotoNagaiPotential(usys=self.usys,
                                           m=1.E11,
                                           a=6.5,
                                           b=0.26)

        # single
        r = [1.,0.,0.]
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        # multiple
        r = np.random.uniform(size=(100,3))
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        grid = np.linspace(-20.,20, 200)
        fig,axes = potential.plot_contours(grid=(grid,0.,grid))
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_2d.png"))

class TestHernquist(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potential = HernquistPotential(usys=self.usys,
                                       m=1.E11, c=10.)

        # single
        r = [1.,0.,0.]
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        # multiple
        r = np.random.uniform(size=(100,3))
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        grid = np.linspace(-20.,20, 50)
        fig,axes = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "hernquist.png"))

class TestLogarithmic(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potentials = []
        potentials.append(LogarithmicPotential(usys=self.usys,
                                                       q1=1., q2=1., q3=1.,
                                                       phi=0., v_c=0.15, r_h=10.))
        potentials.append(LogarithmicPotential(usys=self.usys,
                                                       q1=0.72, q2=1., q3=1.,
                                                       phi=0., v_c=0.15, r_h=1.))
        potentials.append(LogarithmicPotential(usys=self.usys,
                                                       q1=1., q2=0.72, q3=1.,
                                                       phi=np.pi/4, v_c=0.15, r_h=1.))
        potentials.append(LogarithmicPotential(usys=self.usys,
                                                       q1=1., q2=1., q3=0.72,
                                                       phi=0., v_c=0.08, r_h=10.))


        # single
        r = [10.,0.,0.]
        pot_val = potentials[0].value(r)
        acc_val = potentials[0].acceleration(r)

        # multiple
        r = np.random.uniform(10., 50., size=(100,3))
        pot_val = potentials[0].value(r)
        acc_val = potentials[0].acceleration(r)

        grid = np.linspace(-20.,20, 50)

        fig,axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(12,12))

        for ii,potential in enumerate(potentials):
            potential.plot_contours(grid=(grid,grid,0.), ax=axes.flat[ii])

        fig.savefig(os.path.join(plot_path, "log.png"))

class TestNFW(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):

        potentials = []
        potentials.append(NFWPotential(usys=self.usys,
                                       q1=1., q2=1., q3=1.,
                                       v_h=0.15, r_h=10.))
        potentials.append(NFWPotential(usys=self.usys,
                                       q1=0.75, q2=1., q3=1.,
                                       v_h=0.15, r_h=10.))
        potentials.append(NFWPotential(usys=self.usys,
                                       q1=1., q2=0.75, q3=1.,
                                       v_h=0.15, r_h=1.))
        potentials.append(NFWPotential(usys=self.usys,
                                       q1=1., q2=1., q3=1.3,
                                       v_h=0.08, r_h=1.))

        # single
        r = [10.,0.,0.]
        pot_val = potentials[0].value(r)
        acc_val = potentials[0].acceleration(r)

        # multiple
        r = np.random.uniform(10., 50., size=(100,3))
        pot_val = potentials[0].value(r)
        acc_val = potentials[0].acceleration(r)

        grid = np.linspace(-20.,20, 50)

        fig,axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(12,12))

        for ii,potential in enumerate(potentials):
            potential.plot_contours(grid=(grid,grid,0.), ax=axes.flat[ii])

        fig.savefig(os.path.join(plot_path, "nfw.png"))

class TestCompositeGalaxy(object):
    usys = (u.kpc, u.M_sun, u.Myr, u.radian)
    def test_creation(self):
        potential = CompositePotential()
        potential["disk"] = MiyamotoNagaiPotential(usys=self.usys,
                                                   m=1.E11, a=6.5, b=0.26)

        potential["bulge"] = HernquistPotential(usys=self.usys,
                                                m=1.E11, c=0.7)

        potential["halo"] = LogarithmicPotential(usys=self.usys,
                                                         q1=1.4, q2=1., q3=1.5,
                                                         phi=1.69, v_c=0.17, r_h=12.)

        # single
        r = [10.,0.,0.]
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        # multiple
        r = np.random.uniform(10., 50., size=(100,3))
        pot_val = potential.value(r)
        acc_val = potential.acceleration(r)

        grid = np.linspace(-20.,20, 50)
        fig,axes = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "composite_galaxy.png"))
