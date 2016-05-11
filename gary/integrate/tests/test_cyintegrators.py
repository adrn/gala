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
        dp = -p._gradient(w[:3])
        return np.vstack((dq,dp))

    cy_w0 = np.array([[0.,10.,0.,0.2,0.,0.],
                      [10.,0.,0.,0.,0.2,0.]])
    py_w0 = np.ascontiguousarray(cy_w0.T)

    n_steps = 10000
    dt = 2.
    t = np.linspace(0,dt*n_steps,n_steps+1)

    cy_t,cy_w = integrate_func(p.c_instance, cy_w0, t)
    cy_w = np.rollaxis(cy_w, -1)

    integrator = Integrator(F)
    orbit = integrator.run(py_w0, dt=dt, n_steps=n_steps)
    py_t = orbit.t.value
    py_w = orbit.w()

    assert py_w.shape == cy_w.shape
    assert np.allclose(cy_w[:,-1], py_w[:,-1])

@pytest.mark.skipif(True, reason="For timing locally")
def test_time_integration():
    niter = 100

    p = HernquistPotential(m=1E11, c=0.5, units=galactic)

    cy_w0 = np.array([[0.,10.,0.,0.2,0.,0.],
                      [10.,0.,0.,0.,0.2,0.]])

    n_steps = 10000
    dt = 2.
    t = np.linspace(0,dt*n_steps,n_steps+1)

    time0 = time.time()
    for i in range(niter):
        t,w = dop853_integrate_potential(p.c_instance, cy_w0, t)

    exec_time = (time.time()-time0) / niter
    print("{:.3f} per loop iteration".format(exec_time))

@pytest.mark.skipif(True, reason="Slow test - mainly for plotting locally")
@pytest.mark.parametrize(("Integrator","integrate_func"), _list)
def test_scaling(tmpdir, Integrator, integrate_func):
    p = HernquistPotential(m=1E11, c=0.5, units=galactic)

    def F(t,w):
        dq = w[3:]
        dp = -p._gradient(w[:3])
        return np.vstack((dq,dp))

    step_bins = np.logspace(2,np.log10(25000),7)
    colors = ['k', 'b', 'r']
    dt = 1.

    for c,nparticles in zip(colors,[1, 100, 1000]):
        cy_w0 = np.array([[0.,10.,0.,0.2,0.,0.]]*nparticles)
        py_w0 = np.ascontiguousarray(cy_w0.T)

        x = []
        cy_times = []
        py_times = []
        for n_steps in step_bins:
            print(nparticles, n_steps)
            t = np.linspace(0,dt*n_steps,n_steps+1)
            x.append(n_steps)

            # time the Cython integration
            t0 = time.time()
            integrate_func(p.c_instance, cy_w0, t)
            cy_times.append(time.time() - t0)

            # time the Python integration
            t0 = time.time()
            integrator = Integrator(F)
            orbit = integrator.run(py_w0, dt=dt, n_steps=n_steps)
            py_times.append(time.time() - t0)

    #     pl.loglog(x, cy_times, linestyle='-', lw=2., c=c, marker=None,
    #               label="cy: {} orbits".format(nparticles))
    #     pl.loglog(x, py_times, linestyle='--', lw=2., c=c, marker=None,
    #               label="py: {} orbits".format(nparticles))

    # pl.title(Integrator.__name__)
    # pl.legend(loc='upper left')
    # pl.xlim(90,30000)
    # pl.xlabel("N steps")
    # pl.tight_layout()
    # # pl.show()
    # pl.savefig(os.path.join(tmpdir, "integrate-scaling.png"), dpi=300)
