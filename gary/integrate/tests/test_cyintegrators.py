# coding: utf-8
"""
    Test the Cython integrators.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import time

# Third-party
import numpy as np
import matplotlib.pyplot as pl
import pytest

# Project
from ..pyintegrators.leapfrog import LeapfrogIntegrator
from ..cyintegrators.leapfrog import leapfrog_integrate_potential
from ..pyintegrators.dopri853 import DOPRI853Integrator
from ..cyintegrators.dop853 import dop853_integrate_potential
from ...potential import HernquistPotential
from ...units import galactic

integrator_list = [LeapfrogIntegrator, DOPRI853Integrator]
func_list = [leapfrog_integrate_potential, dop853_integrate_potential]
_list = zip(integrator_list, func_list)

# ----------------------------------------------------------------------------

@pytest.mark.parametrize(("Integrator","integrate_func"), _list)
def test_compare_to_py(Integrator, integrate_func):
    p = HernquistPotential(m=1E11, c=0.5, units=galactic)

    def F(t,w):
        dq = w[3:]
        dp = -p.gradient(w[:3])
        return np.vstack((dq,dp))

    cy_w0 = np.array([[0.,10.,0.,0.2,0.,0.],
                      [10.,0.,0.,0.,0.2,0.]])
    py_w0 = np.ascontiguousarray(cy_w0.T)

    nsteps = 10000
    dt = 2.
    t = np.linspace(0,dt*nsteps,nsteps+1)

    cy_t,cy_w = integrate_func(p.c_instance, cy_w0, t)
    cy_w = np.rollaxis(cy_w, -1)

    integrator = Integrator(F)
    py_t,py_w = integrator.run(py_w0, dt=dt, nsteps=nsteps)

    assert py_w.shape == cy_w.shape
    assert np.allclose(cy_w[:,-1], py_w[:,-1])

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
