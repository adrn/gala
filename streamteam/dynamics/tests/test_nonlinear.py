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

def test_gram(self):
    arr = np.array([
      [0.159947293111244, -0.402071263039210, 0.781989928439469, 0.157200868935014],
      [-0.641365729386551, -0.589502248965056, -6.334333712737469E-002, -0.787834065229250],
      [-0.734648540580147, 0.290410423487680, 0.200395494571665, 0.547331391068177],
      [0.140516164496874, -0.649328455579330, -0.621863558066490, 0.402036027551737]])

    fortran_end_arr = np.array([
        [0.176260247376319, -0.443078334791891, 0.861744738228658, 0.173233716603048],
        [-0.539192290034099, -0.517682965825117, -2.242518016864357E-002, -0.663907472888126],
        [-0.823493801368422, 0.237505395299283, 0.194657453290375, 0.477030001352140],
        [7.871383004849420E-003, -0.692298435168163, -0.467976060614355, 0.549235217994234]])

    alf = gram(arr)
    fortran_alf = np.array([0.907449612105405,1.17546413803123,0.974054532627089,0.962464733634354])

    assert np.abs(fortran_alf - alf).sum() < 1E-13
    assert np.abs(arr - fortran_end_arr).sum() < 1E-13

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

        regular_LEs, t, regular_ws = lyapunov(self.regular_w0, self.regular_integrator,
                                              dt=dt, nsteps=nsteps)

        return

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

    def test_simple_lyapunov(self):
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