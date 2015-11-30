# coding: utf-8
"""
    Test the Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third party
import pytest
import numpy as np

# This project
from ..core import *
from ..builtin import *
from ...units import solarsystem

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
        # fig.savefig(os.path.join(plot_path, "composite_kepler_sho_1d.png"))

        fig = potential.plot_contours(grid=(grid,grid,0.))
        # fig.savefig(os.path.join(plot_path, "composite_kepler_sho_2d.png"))
