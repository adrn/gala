# coding: utf-8
"""
    Test the builtin CPotential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.units as u
import pytest

# This project
from ..core import CompositePotential
from ..builtin import *
from ...units import solarsystem, galactic, DimensionlessUnitSystem
from .helpers import PotentialTestBase, CompositePotentialTestBase

##############################################################################
# Python
##############################################################################

class TestHarmonicOscillator1D(PotentialTestBase):
    potential = HarmonicOscillatorPotential(omega=1.)
    w0 = [1.,0.]

    def test_plot(self):
        # Skip for now because contour plotting assumes 3D
        pass

class TestHarmonicOscillator2D(PotentialTestBase):
    potential = HarmonicOscillatorPotential(omega=[1.,2])
    w0 = [1.,0.5,0.,0.1]

    def test_plot(self):
        # Skip for now because contour plotting assumes 3D
        pass

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

class TestIsochroneDimensionless(PotentialTestBase):
    potential = IsochronePotential(units=DimensionlessUnitSystem(), m=1., b=0.1)
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

class TestStone(PotentialTestBase):
    potential = StonePotential(units=galactic, m=1E11, r_c=0.1, r_h=10.)
    w0 = [8.,0.,0.,0.,0.18,0.1]

class TestSphericalNFWPotential(PotentialTestBase):
    potential = SphericalNFWPotential(units=galactic, v_c=0.35, r_s=12.)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

    def test_against_triaxial(self):
        other = LeeSutoTriaxialNFWPotential(units=galactic,
                                            v_c=0.35, r_s=12.,
                                            a=1., b=1., c=1.)

        v1 = other.value(self.w0[:3])
        v2 = self.potential.value(self.w0[:3])
        assert np.allclose(v1.value,v2.value)

        a1 = other.gradient(self.w0[:3])
        a2 = self.potential.gradient(self.w0[:3])
        assert np.allclose(a1.value,a2.value)

    def test_mass_enclosed(self):

        # true mass profile
        vc = self.potential.parameters['v_c'].value
        rs = self.potential.parameters['r_s'].value
        G = self.potential.G

        r = np.linspace(1., 400, 100)
        fac = np.log(1 + r/rs) - (r/rs) / (1 + (r/rs))
        true_mprof = vc**2*rs / (np.log(2)-0.5) / G * fac

        R = np.zeros((3,len(r)))
        R[0,:] = r
        esti_mprof = self.potential.mass_enclosed(R)

        assert np.allclose(true_mprof, esti_mprof.value, rtol=1E-6)

class TestFlattenedNFW(PotentialTestBase):
    potential = FlattenedNFWPotential(units=galactic, v_c=0.2, r_s=12., q_z=0.9)
    w0 = [0.,20.,0.,0.0352238,-0.03579493,0.175]

class TestLeeSutoTriaxialNFW(PotentialTestBase):
    potential = LeeSutoTriaxialNFWPotential(units=galactic, v_c=0.35, r_s=12.,
                                            a=1.3, b=1., c=0.8)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestLogarithmic(PotentialTestBase):
    potential = LogarithmicPotential(units=galactic, v_c=0.17, r_h=10.,
                                     q1=1.2, q2=1., q3=0.8)
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]

class TestComposite(CompositePotentialTestBase):
    p1 = LogarithmicPotential(units=galactic,
                              v_c=0.17, r_h=10.,
                              q1=1.2, q2=1., q3=0.8)
    p2 = MiyamotoNagaiPotential(units=galactic,
                                m=1.E11, a=6.5, b=0.26)
    potential = CompositePotential()
    potential['disk'] = p2
    potential['halo'] = p1
    w0 = [19.0,2.7,-6.9,0.0352238,-0.03579493,0.075]
