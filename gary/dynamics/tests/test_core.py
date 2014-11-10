# coding: utf-8

""" Test core dynamics.  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger

# Project
from ..core import *
from ..plot import plot_orbits
from ...potential import LogarithmicPotential

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_angular_momentum():

    assert np.allclose(angular_momentum([1.,0.,0.,0.,0.,1.]),
                       [0., -1, 0])
    assert np.allclose(angular_momentum([1.,0.,0.,0.,1.,0.]),
                       [0., 0, 1])
    assert np.allclose(angular_momentum([0.,1.,0.,0.,0.,1.]),
                       [1., 0, 0])

def test_classify_orbit():

    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1., phi=0.)

    def make_orbit(x, vx, name):
        # See Binney & Tremaine (2008) Figure 3.8 and 3.9
        E = -0.337
        y = 0.
        vy = np.sqrt(2*(E - potential.value([x,y,0.])))

        w = [x,y,0.,vx,vy,0.]
        t,ws = potential.integrate_orbit(w, dt=0.05, nsteps=10000)
        fig = plot_orbits(ws, linestyle='none', alpha=0.1)
        fig.savefig(os.path.join(plot_path, "{}.png".format(name)))

        return ws

    ws = make_orbit(0.5, 0., "loop")
    loop = classify_orbit(ws)
    assert loop.sum() == 1

    ws = make_orbit(0., 1.5, "box")
    loop = classify_orbit(ws)
    assert loop.sum() == 0