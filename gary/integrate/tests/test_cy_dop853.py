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
import numpy as np
import matplotlib.pyplot as plt

# Project
from ... import potential as gp
from ..dopri853 import DOPRI853Integrator
from ...units import galactic
from ..dopri.wrap_dop853 import dop853_integrate_potential
plot_path = "plots/tests/integrate"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_derp():
    print()
    pot = gp.HernquistPotential(m=1E11, c=0.5, units=galactic)
    # w0 = np.array([1.,2.1,0., 0.,0.5,0.])

    # print("python")
    # print(pot.gradient(w0[:3]))

    # t1 = time.time()
    # t,w = pot.integrate_orbit(w0, dt=0.1, nsteps=10000, Integrator=DOPRI853Integrator)
    # print(w[-1], time.time()-t1)
    # plt.plot(w[:,0,0], w[:,0,1])
    # plt.show()

    nsteps = 100000
    norbitses = 2**np.arange(0,5+1,1)
    times = []
    pytimes = []
    for norbits in norbitses:
        print("{} orbits".format(norbits))
        w0 = np.array([[1.,2.1,0., 0.,0.5,0.]]*norbits)
        t1 = time.time()
        # t,w = dop853_integrate_potential(pot.c_instance, w0, 0.1, 10000, 0., 1E-8, 1E-8)
        t,w = dop853_integrate_potential(pot.c_instance, w0, 0.1, nsteps, 0., 1E-8, 1E-8)
        times.append(time.time()-t1)
        print("cy: {0:.2f}".format(times[-1]))

        # t1 = time.time()
        # t,w = pot.integrate_orbit(w0, dt=0.1, nsteps=nsteps,
        #                           Integrator=DOPRI853Integrator,
        #                           cython_if_possible=False)
        # pytimes.append(time.time()-t1)
        # print("py: {0:.2f}".format(pytimes[-1]))

    from scipy.optimize import leastsq

    def errfunc(p, x, y):
        return y - (p[0]*x + p[1])
    p_opt,ier = leastsq(errfunc, x0=[0.1,0.], args=(norbitses, times))
    print(p_opt)

    plt.plot(norbitses, times)
    # plt.plot(norbitses, pytimes)
    derp = np.linspace(norbitses.min(),norbitses.max(),100)
    plt.plot(derp, p_opt[0]*derp + p_opt[1], marker=None)
    plt.show()

    # plt.figure(figsize=(8,8))
    # plt.plot(w[:,0],w[:,1],marker=None)
    # plt.show()
