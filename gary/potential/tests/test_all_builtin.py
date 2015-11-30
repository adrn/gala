# coding: utf-8
"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np
import astropy.units as u

# This project
from ..core import CompositePotential
from ..builtin import *
from ...units import UnitSystem, solarsystem
from .helpers import PotentialTestBase

# Python
class TestHarmonicOscillator1D(PotentialTestBase):
    potential = HarmonicOscillatorPotential(omega=1.)
    w0 = [1.,0.]

# class TestHarmonicOscillator2D(PotentialTestBase):
#     potential = HarmonicOscillatorPotential(omega=[1.,2])
#     w0 = [1.,0.5,0.,0.1]

# Cython
class TestHenonHeiles(PotentialTestBase):
    def setup(self):
        self.potential = HenonHeilesPotential()
        self.w0 = [1.,0.,0.,0.,2*np.pi,0.]
        super(TestHenonHeiles,self).setup()

class TestKepler(PotentialTestBase):
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

class TestFlattenedNFWPotential(PotentialTestBase):
    def setup(self):
        self.potential = FlattenedNFWPotential(units=self.units,
                                               v_c=0.2, r_s=12., q_z=0.9)
        self.w0 = [0.,20.,0.,0.0352238,-0.03579493,0.175]
        super(TestFlattenedNFWPotential,self).setup()

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

class TestSCFPotential(PotentialTestBase):
    def setup(self):
        self.name = "SCFPotential"
        cc = np.array([[[1.509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-2.606, 0.0, 0.665, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [6.406, 0.0, -0.66, 0.0, 0.044, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-5.5859, 0.0, 0.984, 0.0, -0.03, 0.0, 0.001]], [[-0.086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.221, 0.0, 0.129, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [1.295, 0.0, -0.14, 0.0, -0.012, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.001, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        sc = np.zeros_like(cc)
        self.potential = SCFPotential(m=1E10, r_s=1.,
                                      sin_coeff=sc, cos_coeff=cc,
                                      units=self.units)
        self.w0 = [2.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
        super(TestSCFPotential,self).setup()

class TestWangZhaoBarPotential(PotentialTestBase):
    def setup(self):
        self.name = "WangZhaoBarPotential"
        self.potential = WangZhaoBarPotential(m=1E10, r_s=1.,
                                              alpha=0., Omega=0.,
                                              units=self.units)
        self.w0 = [3.0,0.7,-0.5,0.0352238,-0.03579493,0.075]
        super(TestWangZhaoBarPotential,self).setup()
