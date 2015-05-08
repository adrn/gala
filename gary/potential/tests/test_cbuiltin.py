# coding: utf-8
"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import cPickle as pickle
import os
import time
import numpy as np
import pytest
from astropy.utils.console import color_print
from astropy.constants import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import CompositePotential
from ..cbuiltin import LM10Potential
from ..cbuiltin import *
from ..io import load
from ...units import UnitSystem, galactic, solarsystem

# HACK: bad solution is to do this:
# python setup.py build_ext --inplace

top_path = "plots/"
plot_path = os.path.join(top_path, "tests/potential/cpotential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

units = [u.kpc,u.Myr,u.Msun,u.radian]
G = G.decompose(units)

print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")

niter = 1000
nparticles = 1000

class PotentialTestBase(object):
    name = None
    units = galactic

    def setup(self):
        print("\n\n")
        print("="*50)
        if self.name is None:
            self.name = self.__class__.__name__
        print(self.name)
        self.grid = None

    def test_unitsystem(self):
        assert isinstance(self.potential.units, UnitSystem)

    def test_method_call(self):
        # single
        r = [[1.,0.,0.]]
        pot_val = self.potential.value(r)
        acc_val = self.potential.acceleration(r)

        # multiple
        r = np.random.uniform(size=(nparticles,3))
        pot_val = self.potential.value(r)
        acc_val = self.potential.acceleration(r)

        # save to disk
        self.potential.save("/tmp/potential.yml")
        derp = load("/tmp/potential.yml")

    def test_orbit_integration(self):
        w0 = self.w0
        t1 = time.time()
        t,w = self.potential.integrate_orbit(w0, dt=1., nsteps=10000)
        print("Cython orbit integration time (10000 steps): {}".format(time.time() - t1))

    def test_time_methods(self):

        r = np.random.uniform(size=(nparticles,3))
        for func_name in ["value", "gradient", "acceleration"]:
            t1 = time.time()
            for ii in range(niter):
                x = getattr(self.potential, func_name)(r)
            print("Cython - {}: {:e} sec per call".format(func_name,
                  (time.time()-t1)/float(niter)))

    @pytest.mark.skipif(True, reason="derp.")
    def test_profile(self):
        # Have to turn on cython profiling for this to work
        import pstats, cProfile

        r = np.random.uniform(size=(nparticles,3))
        tmp_value = np.zeros(nparticles)
        tmp_gradient = np.zeros_like(r)
        for func_name in ["value", "gradient"]:
            t1 = time.time()
            for ii in range(niter):
                the_str = "for n in range(10000): self.potential.{}(r)".format(func_name)
                cProfile.runctx(the_str, globals(), locals(), "pro.prof")

                s = pstats.Stats("pro.prof")
                s.strip_dirs().sort_stats("cumulative").print_stats()

                the_str = "for n in range(10000): self.potential.c_instance.{f}(r,tmp_{f})".format(f=func_name)
                cProfile.runctx(the_str, globals(), locals(), "pro.prof")

                s = pstats.Stats("pro.prof")
                s.strip_dirs().sort_stats("cumulative").print_stats()

    def test_plot_contours(self):

        if self.name is None:
            self.name = self.potential.__class__.__name__

        if self.grid is None:
            grid = np.linspace(-50.,50, 200)
        else:
            grid = self.grid

        fig,axes = plt.subplots(1,1)

        t1 = time.time()
        fig = self.potential.plot_contours(grid=(grid,grid,0.),
                                           subplots_kw=dict(figsize=(8,8)))
        print("Cython plot_contours time", time.time() - t1)
        fig.savefig(os.path.join(plot_path, "{}_2d.png"
                    .format(self.name)))

        fig = self.potential.plot_contours(grid=(grid,0,0.),
                                           subplots_kw=dict(figsize=(8,8)))
        fig.savefig(os.path.join(plot_path, "{}_1d_x.png"
                    .format(self.name)))

        fig = self.potential.plot_contours(grid=(0,0,grid),
                                           subplots_kw=dict(figsize=(8,8)))
        fig.savefig(os.path.join(plot_path, "{}_1d_z.png"
                    .format(self.name)))

    def test_pickle(self):
        with open("/tmp/derp.pickle", "w") as f:
            pickle.dump(self.potential, f)

        with open("/tmp/derp.pickle") as f:
            p = pickle.load(f)

        p.value(np.array([[100,0,0.]]))

    def test_mass_enclosed(self):
        r = np.linspace(1., 400, 100)
        R = np.zeros((len(r),3))
        R[:,0] = r
        esti_mprof = self.potential.mass_enclosed(R)

        plt.clf()
        plt.plot(r, esti_mprof)
        plt.savefig(os.path.join(plot_path, "mass_profile_{}.png".format(self.name)))

# ----------------------------------------------------------------------------
#  Potentials to test
#

class TestHenonHeiles(PotentialTestBase):
    units = None

    def setup(self):
        self.grid = np.linspace(-1,1,100)
        self.potential = HenonHeilesPotential()
        self.w0 = [1.,0.,0.,0.,2*np.pi,0.]
        super(TestHenonHeiles,self).setup()

class TestKepler(PotentialTestBase):
    units = solarsystem

    def setup(self):
        self.potential = KeplerPotential(units=self.units, m=1.)
        self.w0 = [1.,0.,0.,0.,2*np.pi,0.]
        super(TestKepler,self).setup()

class TestIsochrone(PotentialTestBase):
    units = solarsystem

    def setup(self):
        self.potential = IsochronePotential(units=self.units, m=1., b=0.1)
        self.w0 = [1.,0.,0.,0.,2*np.pi,0.]
        super(TestIsochrone,self).setup()

class TestIsochroneU(PotentialTestBase):
    units = (u.yr, u.au, u.Msun, u.radian)

    def setup(self):
        self.potential = IsochronePotential(units=self.units, m=1., b=0.1)
        self.w0 = [1.,0.,0.,0.,2*np.pi,0.]
        super(TestIsochroneU,self).setup()

class TestHernquist(PotentialTestBase):
    def setup(self):
        self.potential = HernquistPotential(units=self.units,
                                            m=1.E11, c=0.26)
        self.w0 = [1.,0.,0.,0.,0.1,0.1]
        super(TestHernquist,self).setup()

class TestPlummer(PotentialTestBase):
    def setup(self):
        self.potential = PlummerPotential(units=self.units,
                                          m=1.E11, b=0.26)
        self.w0 = [1.,0.,0.,0.,0.1,0.1]
        super(TestPlummer,self).setup()

class TestJaffe(PotentialTestBase):
    def setup(self):
        self.potential = JaffePotential(units=self.units,
                                        m=1.E11, c=0.26)
        self.w0 = [1.,0.,0.,0.,0.1,0.1]
        super(TestJaffe,self).setup()

class TestMiyamotoNagai(PotentialTestBase):
    def setup(self):
        self.potential = MiyamotoNagaiPotential(units=self.units,
                                                m=1.E11, a=6.5, b=0.26)
        self.w0 = [8.,0.,0.,0.,0.22,0.1]
        super(TestMiyamotoNagai,self).setup()

# class TestStone(PotentialTestBase):
#     units = galactic

#     def setup(self):
#         self.potential = StonePotential(units=self.units,
#                                         v_c=0.2, r_c=1, r_t=2.)
#         self.w0 = [8.,0.,0.,0.,0.22,0.1]
#         super(TestStone,self).setup()

class TestSphericalNFWPotential(PotentialTestBase):
    def setup(self):
        self.potential = SphericalNFWPotential(units=self.units,
                                               v_c=0.35,  # np.sqrt(np.log(2)-0.5)
                                               r_s=12.)
        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestSphericalNFWPotential,self).setup()

    def test_against_triaxial(self):
        other = LeeSutoTriaxialNFWPotential(units=self.units,
                                            v_c=0.35, r_s=12.,
                                            a=1., b=1., c=1.)

        v1 = other.value(np.array([self.w0[:3]]))
        v2 = self.potential.value(np.array([self.w0[:3]]))
        assert np.allclose(v1,v2)

        a1 = other.acceleration(np.array([self.w0[:3]]))
        a2 = self.potential.acceleration(np.array([self.w0[:3]]))
        assert np.allclose(a1,a2)

    def test_mass_enclosed(self):

        # true mass profile
        vc = self.potential.parameters['v_c']
        rs = self.potential.parameters['r_s']
        G = self.potential.G

        r = np.linspace(1., 400, 100)
        fac = np.log(1 + r/rs) - (r/rs) / (1 + (r/rs))
        true_mprof = vc**2*rs / (np.log(2)-0.5) / G * fac

        R = np.zeros((len(r),3))
        R[:,0] = r
        esti_mprof = self.potential.mass_enclosed(R)

        plt.clf()
        plt.plot(r, true_mprof, label='true')
        plt.plot(r, esti_mprof, label='estimated')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(plot_path, "mass_profile_nfw.png"))

        assert np.allclose(true_mprof, esti_mprof, rtol=1E-6)

class TestLeeSutoTriaxialNFWPotential(PotentialTestBase):
    def setup(self):
        self.potential = LeeSutoTriaxialNFWPotential(units=self.units,
                                                     v_c=0.35, r_s=12.,
                                                     a=1.3, b=1., c=0.8)
        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestLeeSutoTriaxialNFWPotential,self).setup()

class TestMisalignedLeeSutoNFWPotential(PotentialTestBase):
    def setup(self):
        self.potential = LeeSutoTriaxialNFWPotential(units=self.units,
                                                     v_c=0.35, r_s=12.,
                                                     a=1.4, b=1., c=0.6,
                                                     phi=np.radians(30.),
                                                     theta=np.radians(30))
        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestMisalignedLeeSutoNFWPotential,self).setup()
        self.name = "MisalignedLeeSutoNFWPotential"

class TestLogarithmicPotential(PotentialTestBase):
    def setup(self):
        self.potential = LogarithmicPotential(units=self.units,
                                              v_c=0.17, r_h=10.,
                                              q1=1.2, q2=1., q3=0.8)
        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestLogarithmicPotential,self).setup()

class TestMisalignedLogarithmicPotential(PotentialTestBase):
    def setup(self):
        self.name = "MisalignedLogarithmicPotential"
        self.potential = LogarithmicPotential(units=self.units,
                                              v_c=0.17, r_h=10.,
                                              q1=1.2, q2=1., q3=0.8, phi=0.35)
        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestMisalignedLogarithmicPotential,self).setup()

class TestLM10Potential(PotentialTestBase):
    def setup(self):
        self.name = "LM10Potential"
        self.potential = LM10Potential(units=self.units)
        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestLM10Potential,self).setup()

class TestCompositePotential(PotentialTestBase):
    def setup(self):
        self.name = "CompositePotential"
        p1 = LogarithmicPotential(units=self.units,
                                  v_c=0.17, r_h=10.,
                                  q1=1.2, q2=1., q3=0.8, phi=0.35)
        p2 = MiyamotoNagaiPotential(units=self.units,
                                    m=1.E11, a=6.5, b=0.26)
        self.potential = CompositePotential(disk=p2, halo=p1)

        self.w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

        super(TestCompositePotential,self).setup()
