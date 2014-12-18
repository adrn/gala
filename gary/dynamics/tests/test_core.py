# coding: utf-8

""" Test core dynamics.  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger

# Project
from ..core import *
from ..plot import plot_orbits
from ...potential import LogarithmicPotential
from ...units import galactic

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# ----------------------------------------------------------------------------

def test_angular_momentum():

    assert np.allclose(angular_momentum([1.,0.,0.,0.,0.,1.]),
                       [0., -1, 0])
    assert np.allclose(angular_momentum([1.,0.,0.,0.,1.,0.]),
                       [0., 0, 1])
    assert np.allclose(angular_momentum([0.,1.,0.,0.,0.,1.]),
                       [1., 0, 0])

# ----------------------------------------------------------------------------

def make_known_orbit(x, vx, potential, name):
    # See Binney & Tremaine (2008) Figure 3.8 and 3.9
    E = -0.337
    y = 0.
    vy = np.sqrt(2*(E - potential.value([x,y,0.])))

    w = [x,y,0.,vx,vy,0.]
    t,ws = potential.integrate_orbit(w, dt=0.05, nsteps=10000)
    fig = plot_orbits(ws, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(plot_path, "{}.png".format(name)))

    return ws

def test_classify_orbit():

    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1.,
                                     units=galactic)
    ws = make_known_orbit(0.5, 0., potential, "loop")
    loop = classify_orbit(ws)
    assert loop.sum() == 1

    ws = make_known_orbit(0., 1.5, potential, "box")
    loop = classify_orbit(ws)
    assert loop.sum() == 0

# ----------------------------------------------------------------------------

def test_align_circulation():

    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1.,
                                     units=galactic)
    w0 = np.array([[0.,1.,0.,0.,0.,0.5],  # loop around x axis
                   [1.,0.,0.,0.,0.,0.5],  # loop around y axis
                   [1.,0.,0.,0.,0.5,0.],  # loop around z axis
                   [0.8,0.4,0.,0.,0.1,0.]])  # box
    names = ['xloop', 'yloop', 'zloop', 'box']

    t,w = potential.integrate_orbit(w0, dt=0.05, nsteps=10000)
    fig = plot_orbits(w, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(plot_path, "align_circulation_orbits_init.png"))

    circ = classify_orbit(w)
    new_w = align_circulation_with_z(w, circ)
    fig = plot_orbits(new_w, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(plot_path, "align_circulation_orbits_post.png"))

    new_circ = classify_orbit(new_w)
    assert np.all(new_circ[:3,2] == 1.)
