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
from ...units import solarsystem, galactic

top_path = "plots/"
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

class TestComposite(object):
    units = solarsystem

    def test_composite_create(self):
        potential = CompositePotential()

        # Add a point mass with same unit system
        potential["one"] = KeplerPotential(units=self.units, m=1.)

        with pytest.raises(TypeError):
            potential["two"] = "derp"

        assert "one" in potential.parameters
        assert "m" in potential.parameters["one"]
        with pytest.raises(TypeError):
            potential.parameters["m"] = "derp"

    def test_plot_composite(self):
        potential = CompositePotential()

        # Add a kepler potential and a harmonic oscillator
        potential["one"] = KeplerPotential(m=1., units=self.units)
        potential["two"] = HarmonicOscillatorPotential(omega=[0.1,0.2,0.31],
                                                       units=self.units)

        grid = np.linspace(-5.,5)
        fig = potential.plot_contours(grid=(grid,0.,0.))
        fig.savefig(os.path.join(plot_path, "composite_kepler_sho_1d.png"))

        fig = potential.plot_contours(grid=(grid,grid,0.))
        fig.savefig(os.path.join(plot_path, "composite_kepler_sho_2d.png"))
