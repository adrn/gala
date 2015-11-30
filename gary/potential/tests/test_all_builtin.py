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
from ...units import solarsystem, galactic
from .helpers import PotentialTestBase

##############################################################################
# Python
##############################################################################

class TestHarmonicOscillator1D(PotentialTestBase):
    potential = HarmonicOscillatorPotential(omega=1.)
    w0 = [1.,0.]

# class TestHarmonicOscillator2D(PotentialTestBase):
#     potential = HarmonicOscillatorPotential(omega=[1.,2])
#     w0 = [1.,0.5,0.,0.1]

##############################################################################
# Cython
##############################################################################

class TestHenonHeiles(PotentialTestBase):
    potential = HenonHeilesPotential()
    w0 = [1.,0.,0.,0.,2*np.pi,0.]

class TestKepler(PotentialTestBase):
    potential = KeplerPotential(units=solarsystem, m=1.)
    w0 = [1.,0.,0.,0.,2*np.pi,0.]

class TestKeplerUnitInput(PotentialTestBase):
    potential = KeplerPotential(units=solarsystem, m=1E-3*u.ksolMass)
    w0 = [1.,0.,0.,0.,2*np.pi,0.]

class TestIsochrone(PotentialTestBase):
    potential = IsochronePotential(units=solarsystem, m=1., b=0.1)
    w0 = [1.,0.,0.,0.,2*np.pi,0.]

class TestHernquist(PotentialTestBase):
    potential = HernquistPotential(units=galactic, m=1.E11, c=0.26)
    w0 = [1.,0.,0.,0.,0.1,0.1]

class TestPlummer(PotentialTestBase):
    potential = PlummerPotential(units=galactic, m=1.E11, b=0.26)
    w0 = [1.,0.,0.,0.,0.1,0.1]

class TestJaffe(PotentialTestBase):
    potential = JaffePotential(units=galactic, m=1.E11, c=0.26)
    w0 = [1.,0.,0.,0.,0.1,0.1]

class TestMiyamotoNagai(PotentialTestBase):
    potential = MiyamotoNagaiPotential(units=galactic, m=1.E11, a=6.5, b=0.26)
    w0 = [8.,0.,0.,0.,0.22,0.1]

# class TestStone(PotentialTestBase):
#     potential = StonePotential(units=galactic, v_c=0.2, r_c=1, r_t=2.)
#     w0 = [8.,0.,0.,0.,0.22,0.1]

class TestSphericalNFWPotential(PotentialTestBase):
    potential = SphericalNFWPotential(units=galactic, v_c=0.35, r_s=12.)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

    def test_against_triaxial(self):
        other = LeeSutoTriaxialNFWPotential(units=galactic,
                                            v_c=0.35, r_s=12.,
                                            a=1., b=1., c=1.)

        v1 = other.value(self.w0[:3])
        v2 = self.potential.value(self.w0[:3])
        assert np.allclose(v1,v2)

        a1 = other.gradient(self.w0[:3])
        a2 = self.potential.gradient(self.w0[:3])
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
    potential = FlattenedNFWPotential(units=galactic, v_c=0.2, r_s=12., q_z=0.9)
    w0 = [0.,20.,0.,0.0352238,-0.03579493,0.175]

class TestLeeSutoTriaxialNFWPotential(PotentialTestBase):
    potential = LeeSutoTriaxialNFWPotential(units=galactic, v_c=0.35, r_s=12.,
                                            a=1.3, b=1., c=0.8)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestMisalignedLeeSutoNFWPotential(PotentialTestBase):
    potential = LeeSutoTriaxialNFWPotential(units=galactic, v_c=0.35, r_s=12.,
                                            a=1.4, b=1., c=0.6,
                                            phi=30.*u.deg, theta=30*u.deg)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestLogarithmicPotential(PotentialTestBase):
    potential = LogarithmicPotential(units=galactic, v_c=0.17, r_h=10.,
                                     q1=1.2, q2=1., q3=0.8)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestMisalignedLogarithmicPotential(PotentialTestBase):
    potential = LogarithmicPotential(units=galactic, v_c=0.17, r_h=10.,
                                     q1=1.2, q2=1., q3=0.8, phi=41*u.deg)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestLM10Potential(PotentialTestBase):
    potential = LM10Potential(units=galactic)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestCompositePotential(PotentialTestBase):
    p1 = LogarithmicPotential(units=galactic,
                              v_c=0.17, r_h=10.,
                              q1=1.2, q2=1., q3=0.8, phi=0.35)
    p2 = MiyamotoNagaiPotential(units=galactic,
                                m=1.E11, a=6.5, b=0.26)
    potential = CompositePotential()
    potential['disk'] = p2
    potential['halo'] = p1
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

# class TestSCFPotential(PotentialTestBase):
#     cc = np.array([[[1.509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-2.606, 0.0, 0.665, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [6.406, 0.0, -0.66, 0.0, 0.044, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-5.5859, 0.0, 0.984, 0.0, -0.03, 0.0, 0.001]], [[-0.086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.221, 0.0, 0.129, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [1.295, 0.0, -0.14, 0.0, -0.012, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.001, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
#     sc = np.zeros_like(cc)
#     potential = SCFPotential(m=1E10, r_s=1.,
#                              sin_coeff=sc, cos_coeff=cc,
#                              units=galactic)
#     w0 = [2.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

# class TestWangZhaoBarPotential(PotentialTestBase):
#     potential = WangZhaoBarPotential(m=1E10, r_s=1., alpha=0., Omega=0.,
#                                      units=galactic)
#     w0 = [3.0,0.7,-0.5,0.0352238,-0.03579493,0.075]
