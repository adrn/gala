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
from ..nonlinear import lyapunov, fft_orbit
from ...integrate import DOPRI853Integrator

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestForcedPendulum(object):

    def setup(self):
        def F(t,x,A,omega_d):
            q,p = x.T
            return np.array([p,-np.sin(q) + A*np.cos(omega_d*t)]).T

        # initial conditions and parameter choices for chaotic / regular pendulum
        self.regular_w0 = np.array([1.,0.])
        self.regular_par = (0.055, 0.7)
        self.regular_integrator = DOPRI853Integrator(F, func_args=self.regular_par)

        self.chaotic_w0 = np.array([3.,0.])
        self.chaotic_par = (0.07, 0.75)
        self.chaotic_integrator = DOPRI853Integrator(F, func_args=self.chaotic_par)

    def test_lyapunov(self):
        nsteps = 10000
        dt = 0.1
        nsteps_per_pullback = 10
        d0 = 1e-5
        noffset = 10

        regular_LEs, t, regular_ws = lyapunov(self.regular_w0, self.regular_integrator,
                                              dt=dt, nsteps=nsteps,
                                              d0=d0, nsteps_per_pullback=nsteps_per_pullback,
                                              noffset=noffset)

        chaotic_LEs, t, chaotic_ws = lyapunov(self.chaotic_w0, self.chaotic_integrator,
                                              dt=dt, nsteps=nsteps,
                                              d0=d0, nsteps_per_pullback=nsteps_per_pullback,
                                              noffset=noffset)
        plt.clf()
        plt.semilogy(regular_LEs, marker=None)
        plt.semilogy(np.mean(regular_LEs,axis=1), marker=None, linewidth=2.)
        plt.savefig(os.path.join(plot_path,"pend_le_regular.png"))

        plt.clf()
        plt.plot(t, regular_ws[:,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_orbit_regular.png"))

        plt.clf()
        plt.semilogy(chaotic_LEs, marker=None)
        plt.semilogy(np.mean(chaotic_LEs,axis=1), marker=None, linewidth=2)
        plt.savefig(os.path.join(plot_path,"pend_le_chaotic.png"))

        plt.clf()
        plt.plot(t, chaotic_ws[:,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_orbit_chaotic.png"))

    def test_frequency(self):
        import scipy.signal as ss
        nsteps = 10000
        dt = 0.1

        t,w = self.regular_integrator.run(self.regular_w0, dt=dt, nsteps=nsteps)
        f,fft = fft_orbit(t, w)

        peak_ix = ss.find_peaks_cwt(fft[:,0], widths=np.linspace(dt*2, dt*100, 10))
        print(peak_ix)

        plt.clf()
        plt.axvline(self.regular_par[1]/(2*np.pi), linewidth=3., alpha=0.35, color='b')
        plt.axvline(1/(2*np.pi), linewidth=3., alpha=0.35, color='r')
        plt.semilogx(f[:,0], fft[:,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_fft_regular.png"))

        # ----------------------------------------------------------------------
        t,w = self.chaotic_integrator.run(self.chaotic_w0, dt=dt, nsteps=nsteps)
        f,fft = fft_orbit(t, w)

        peak_ix = ss.find_peaks_cwt(fft[:,0], widths=np.linspace(dt*2, dt*100, 10))
        print(peak_ix)

        plt.clf()
        plt.axvline(self.chaotic_par[1]/(2*np.pi), linewidth=3., alpha=0.35, color='b')
        plt.axvline(1/(2*np.pi), linewidth=3., alpha=0.35, color='r')
        plt.semilogx(f[:,0], fft[:,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_fft_chaotic.png"))