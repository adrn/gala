# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Project
from ..nonlinear import lyapunov
from ...integrate import DOPRI853Integrator

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestLyapunov(object):

    def test_pendulum(self):
        # forced pendulum

        def F(t,x,A,omega_d):
            q,p = x.T
            return np.array([p,-np.sin(q) + A*np.cos(omega_d*t)]).T

        # initial conditions and parameter choices for chaotic / regular pendulum
        regular_w0 = np.array([1.,0.])
        regular_integrator = DOPRI853Integrator(F, func_args=(0.055, 0.7))

        chaotic_w0 = np.array([3.,0.])
        chaotic_integrator = DOPRI853Integrator(F, func_args=(0.07, 0.75))

        plain_integrator = DOPRI853Integrator(F, func_args=(0.07, 0.75))

        nsteps = 10000
        dt = 0.1
        nsteps_per_pullback = 10
        d0 = 1e-5
        noffset = 10

        regular_LEs, regular_ws = lyapunov(regular_w0, regular_integrator, dt, nsteps,
                                           d0=d0, nsteps_per_pullback=nsteps_per_pullback,
                                           noffset=noffset)

        chaotic_LEs, chaotic_ws = lyapunov(chaotic_w0, chaotic_integrator, dt=dt, nsteps=nsteps,
                                           d0=d0, nsteps_per_pullback=nsteps_per_pullback,
                                           noffset=noffset)

        ts, ws = plain_integrator.run(chaotic_w0, dt=dt, nsteps=nsteps)

        plt.clf()
        plt.semilogy(regular_LEs, marker=None)
        plt.semilogy(np.mean(regular_LEs,axis=1), marker=None, linewidth=2.)
        plt.savefig(os.path.join(plot_path,"pend_le_regular.png"))

        plt.clf()
        plt.plot(regular_ws[:,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_orbit_regular.png"))

        plt.clf()
        plt.semilogy(chaotic_LEs, marker=None)
        plt.semilogy(np.mean(chaotic_LEs,axis=1), marker=None, linewidth=2)
        plt.savefig(os.path.join(plot_path,"pend_le_chaotic.png"))

        plt.clf()
        plt.plot(chaotic_ws[:,0], marker=None)
        plt.plot(ws[:,0,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_orbit_chaotic.png"))