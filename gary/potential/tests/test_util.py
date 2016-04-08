# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# This project
from ...units import solarsystem
from ..util import from_equation
from .helpers import PotentialTestBase

##############################################################################
# Python
##############################################################################

class TestHarmonicOscillatorFromEquation(PotentialTestBase):
    Potential = from_equation("1/2*k*x**2", vars="x", pars="k",
                              name='HarmonicOscillator')
    potential = Potential(k=1.)
    w0 = [1.,0.]

    def test_plot(self):
        # Skip for now because contour plotting assumes 3D
        pass

    def test_pickle(self):
        # Skip for now because these are not picklable
        pass

class TestHarmonicOscillatorFromEquationUnits(PotentialTestBase):
    Potential = from_equation("1/2*k*x**2", vars="x", pars="k",
                              name='HarmonicOscillator')
    potential = Potential(k=1., units=solarsystem)
    w0 = [1.,0.]

    def test_plot(self):
        # Skip for now because contour plotting assumes 3D
        pass

    def test_pickle(self):
        # Skip for now because these are not picklable
        pass
