# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cStringIO as stringio

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Project
from ..nonlinear import lyapunov_max, lyapunov_spectrum, sali, fft_orbit
from ...integrate import DOPRI853Integrator
from ...util import gram_schmidt

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_gram():
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

    alf = gram_schmidt(arr)
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

    def test_lyapunov_max(self):
        nsteps = 100000
        dt = 0.1
        nsteps_per_pullback = 10
        d0 = 1e-5
        noffset = 8

        regular_LEs, t, regular_ws = lyapunov_max(self.regular_w0, self.regular_integrator,
                                                  dt=dt, nsteps=nsteps,
                                                  d0=d0, nsteps_per_pullback=nsteps_per_pullback,
                                                  noffset=noffset)
        regular_LEs = np.mean(regular_LEs, axis=1)

        chaotic_LEs, t, chaotic_ws = lyapunov_max(self.chaotic_w0, self.chaotic_integrator,
                                                  dt=dt, nsteps=nsteps,
                                                  d0=d0, nsteps_per_pullback=nsteps_per_pullback,
                                                  noffset=noffset)
        chaotic_LEs = np.mean(chaotic_LEs, axis=1)

        plt.clf()
        plt.loglog(regular_LEs, marker=None)
        plt.savefig(os.path.join(plot_path,"pend_regular.png"))

        plt.clf()
        plt.plot(t, regular_ws[:,0], marker=None)
        plt.savefig(os.path.join(plot_path,"pend_orbit_regular.png"))

        plt.clf()
        plt.loglog(chaotic_LEs, marker=None)
        plt.savefig(os.path.join(plot_path,"pend_chaotic.png"))

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

# --------------------------------------------------------------------

def potential(w,A,B,C,D):
    x,y = w[...,:2].T
    term1 = 0.5*(A*x**2 + B*y**2)
    term2 = D*x**2*y - C/3.*y**3
    return term1 + term2

def acceleration(w,A,B,C,D):
    x,y = w[...,:2].T
    ax = -(A*x + 2*D*x*y)
    ay = -(B*y + D*x*x - C*y*y)
    return np.array([ax, ay]).T

def jerk(w,A,B,C,D):
    x,y = w[...,:2].T
    dx,dy = w[...,4:6].T

    dax = -(A+2*D*y)*dx - 2*D*x*dy
    day = -2*D*x*dx - (B-2*C*y)*dy

    return np.array([dax,day]).T

def F_max(t,w,*args):
    x,y,px,py = w.T
    term1 = np.array([px, py]).T
    term2 = acceleration(w, *args)

    #print(term1, term2)
    return np.hstack((term1,term2))

def F_spec(t,w,*args):
    x,y,px,py,dx,dy,dpx,dpy = w.T
    term1 = np.array([px, py]).T
    term2 = acceleration(w, *args)
    term3 = np.array([dpx,dpy]).T
    term4 = jerk(w, *args)
    return np.hstack((term1,term2,term3,term4))

class TestHenonHeiles(object):

    def setup(self):
        # parameter choices
        self.par = (1.,1.,1.,1.)

        # initial conditions from LP-VI documentation:
        self.w0s = np.array([[0.,0.295456,0.407308431,0.], # stable, period 1
                             [0., 0.483, 0.27898039, 0.], # stable, quasi-periodic
                             [0., 0.46912, 0.291124891, 0.], # quasi-periodic, close to unstable
                             [0., 0.509, 0.254624859, 0.], # irregular
                             [0., 0.56, 0.164113781, 0.112]]) # irregular

        self.nsteps = 10000
        self.dt = 0.1

    def test_orbit(self):
        integrator = DOPRI853Integrator(F_max, func_args=self.par)

        plt.clf()
        for ii,w0 in enumerate(self.w0s):
            t,w = integrator.run(w0, dt=self.dt, nsteps=self.nsteps)
            plt.plot(w[:,0,0], w[:,0,1], marker=None)

        plt.savefig(os.path.join(plot_path,"hh_orbits.png"))

    def test_lyapunov_max(self):
        nsteps_per_pullback = 10
        d0 = 1e-5
        noffset = 8

        integrator = DOPRI853Integrator(F_max, func_args=self.par)
        for ii,w0 in enumerate(self.w0s):
            lyap, t, ws = lyapunov_max(w0, integrator,
                                       dt=self.dt, nsteps=self.nsteps,
                                       d0=d0, noffset=noffset,
                                       nsteps_per_pullback=nsteps_per_pullback)
            lyap = np.mean(lyap, axis=1)

            plt.clf()
            plt.loglog(lyap, marker=None)
            plt.savefig(os.path.join(plot_path,"hh_lyap_max_{}.png".format(ii)))

            plt.clf()
            plt.plot(ws[...,0], ws[...,1], marker=None)
            plt.savefig(os.path.join(plot_path,"hh_orbit_lyap_max_{}.png".format(ii)))

    def test_lyapunov_spectrum(self):

        integrator = DOPRI853Integrator(F_spec, func_args=self.par)
        for ii,w0 in enumerate(self.w0s):
            lyap, t, ws = lyapunov_spectrum(w0, integrator,
                                            dt=self.dt, nsteps=self.nsteps)

            plt.clf()
            plt.loglog(lyap, marker=None)
            plt.savefig(os.path.join(plot_path,"hh_lyap_spec_{}.png".format(ii)))

            plt.clf()
            plt.plot(ws[...,0], ws[...,1], marker=None)
            plt.savefig(os.path.join(plot_path,"hh_orbit_lyap_spec_{}.png".format(ii)))

    def test_sali(self):
        tbls = []
        with get_pkg_data_fileobj('hh.sali') as f:
            d = f.read()
            blocks = d.split("\n\n")

            for block in blocks:
                tbls.append(np.loadtxt(stringio.StringIO(block)))

        integrator = DOPRI853Integrator(F_spec, func_args=self.par)
        for ii,w0 in enumerate(self.w0s):
            s, t, ws = sali(w0, integrator, dt=self.dt, nsteps=self.nsteps)

            plt.clf()
            plt.loglog(t, s, marker=None)
            plt.loglog(tbls[ii][:,0], tbls[ii][:,1], marker=None, alpha=0.4)
            plt.savefig(os.path.join(plot_path,"hh_sali_{}.png".format(ii)))

            plt.clf()
            plt.plot(ws[...,0], ws[...,1], marker=None)
            plt.savefig(os.path.join(plot_path,"hh_orbit_sali_{}.png".format(ii)))
