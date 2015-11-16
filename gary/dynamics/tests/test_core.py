# coding: utf-8

""" Test core dynamics.  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging

# Third-party
import astropy.units as u
import numpy as np
from astropy import log as logger

# Project
from ..core import *
from ..plot import plot_orbits
from ...potential import LogarithmicPotential
from ...units import galactic

logger.setLevel(logging.DEBUG)

# ----------------------------------------------------------------------------

def test_angular_momentum():

    assert np.allclose(angular_momentum([1.,0.,0.],[0.,0.,1.]),
                       [0., -1, 0])
    assert np.allclose(angular_momentum([1.,0.,0.],[0.,1.,0.]),
                       [0., 0, 1])
    assert np.allclose(angular_momentum([0.,1.,0.],[0.,0.,1.]),
                       [1., 0, 0])

    q = [1.,0,0]*u.kpc
    p = [0,200.,0]*u.pc/u.Myr
    np.testing.assert_allclose(angular_momentum(q,p).to(u.kpc**2/u.Myr),
                               [0,0,0.2]*u.kpc**2/u.Myr)

# ----------------------------------------------------------------------------

def make_known_orbit(tmpdir, x, vx, potential, name):
    # See Binney & Tremaine (2008) Figure 3.8 and 3.9
    E = -0.337
    y = 0.
    vy = np.sqrt(2*(E - potential.value([x,y,0.])))[0]

    w = [x,y,0.,vx,vy,0.]
    t,ws = potential.integrate_orbit(w, dt=0.05, nsteps=10000)
    fig = plot_orbits(ws, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(str(tmpdir), "{}.png".format(name)))

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

    # try also for a single orbit
    loop = classify_orbit(ws[:,0])
    assert loop.shape == (3,)
    assert loop.sum() == 0

# ----------------------------------------------------------------------------

def test_align_circulation_single():

    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1.,
                                     units=galactic)
    w0 = np.array([[0.,1.,0.,0.,0.,0.5],  # loop around x axis
                   [1.,0.,0.,0.,0.,0.5],  # loop around y axis
                   [1.,0.,0.,0.,0.5,0.],  # loop around z axis
                   [0.8,0.4,0.,0.,0.1,0.]])  # box

    t,w = potential.integrate_orbit(w0, dt=0.05, nsteps=10000)

    for i in range(w.shape[1]):
        circ = classify_orbit(w[:,i])
        new_w = align_circulation_with_z(w[:,i], circ)
        new_circ = classify_orbit(new_w)

        if i == 3:
            assert np.sum(new_circ) == 0
        else:
            assert new_circ[2] == 1.

def test_align_circulation_many(tmpdir):

    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1.,
                                     units=galactic)
    w0 = np.array([[0.,1.,0.,0.,0.,0.5],  # loop around x axis
                   [1.,0.,0.,0.,0.,0.5],  # loop around y axis
                   [1.,0.,0.,0.,0.5,0.],  # loop around z axis
                   [0.8,0.4,0.,0.,0.1,0.]])  # box
    names = ['xloop', 'yloop', 'zloop', 'box']

    t,w = potential.integrate_orbit(w0, dt=0.05, nsteps=10000)
    fig = plot_orbits(w, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(str(tmpdir), "align_circulation_orbits_init.png"))

    circ = classify_orbit(w)
    assert circ.shape == (4,3)

    new_w = align_circulation_with_z(w, circ)
    fig = plot_orbits(new_w, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(str(tmpdir), "align_circulation_orbits_post.png"))

    new_circ = classify_orbit(new_w)
    assert np.all(new_circ[:3,2] == 1.)
