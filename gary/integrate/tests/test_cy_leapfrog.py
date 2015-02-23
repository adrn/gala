# coding: utf-8
"""
    Test the integrators.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import time

# Third-party
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

# Project
from .._leapfrog import cy_leapfrog_run
from ...potential import HernquistPotential
from ...units import galactic

plot_path = "plots/tests/integrate"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_py_compare():
    p = HernquistPotential(m=1E11, c=0.5, units=galactic)

    # w0 = np.array([[0.,1.,0.,0.2,0.,0.],
    #                [1.,0.,0.,0.,0.2,0.]])
    w0 = np.array([[0.,1.,0.,0.2,0.,0.]]*1)
    nsteps = 10000

    cy_t,cy_w = cy_leapfrog_run(p.c_instance, w0, 0.1, nsteps, 0.)
    py_t,py_w = p.integrate_orbit(w0, dt=0.1, nsteps=nsteps)

    np.testing.assert_allclose(cy_w[-1], py_w[-1])

def test_scaling():
    p = HernquistPotential(m=1E11, c=0.5, units=galactic)

    step_bins = np.logspace(2,np.log10(25000),7)
    colors = ['k', 'b', 'r']

    for c,nparticles in zip(colors,[1, 100, 1000]):
        w0 = np.array([[0.,1.,0.,0.2,0.,0.]]*nparticles)
        x = []
        y = []
        for nsteps in step_bins:
            print(nparticles, nsteps)
            t1 = time.time()
            cy_leapfrog_run(p.c_instance, w0, 0.1, nsteps, 0.)
            x.append(nsteps)
            y.append(time.time() - t1)

        plt.loglog(x, y, linestyle='-', lw=2., c=c)

    for c,nparticles in zip(colors,[1, 100, 1000]):
        w0 = np.array([[0.,1.,0.,0.2,0.,0.]]*nparticles)
        x = []
        y = []
        for nsteps in step_bins:
            print(nparticles, nsteps)
            t1 = time.time()
            p.integrate_orbit(w0, dt=0.1, nsteps=nsteps)
            x.append(nsteps)
            y.append(time.time() - t1)

        plt.loglog(x, y, linestyle='--', lw=2., c=c)

    plt.xlim(95,10100)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "cy-scaling.png"))
