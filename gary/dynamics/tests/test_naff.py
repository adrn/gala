# coding: utf-8

""" Test action-angle stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import argrelmax

# Project
from ..naff import NAFF
from ...integrate import DOPRI853Integrator
from ... import potential as gp
from ... import dynamics as gd
from ...units import galactic

plot_path = "plots/tests/dynamics/naff"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

logger.important("Plots located at: {}".format(plot_path))

def estimate_axisym_freqs(t, w):
    R = np.sqrt(w[:,0,0]**2 + w[:,0,1]**2)
    phi = np.arctan2(w[:,0,1], w[:,0,0])
    z = w[:,0,2]

    ix = argrelmax(R)[0]
    fR = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))
    ix = argrelmax(phi)[0]
    fphi = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))
    ix = argrelmax(z)[0]
    fz = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))

    return fR, fphi, fz

class NAFFBase(object):

    def setup(self):
        self.name = self.__class__.__name__

        logger.debug("-"*50)
        logger.debug(self.name)

        logger.debug("dt : {}".format(self.dt))
        logger.debug("Nsteps : {}".format(self.nsteps))

        self.filename = os.path.join(plot_path,"orbit_{}.npy".format(self.name))
        if not os.path.exists(self.filename):
            t,w = self.potential.integrate_orbit(self.w0, dt=self.dt,
                                                 nsteps=self.nsteps,
                                                 Integrator=DOPRI853Integrator)
            np.save(self.filename, np.vstack((w.T,t[np.newaxis,np.newaxis])))
        wt = np.load(self.filename)
        self.w = wt[:6].T.copy()
        self.t = wt[6,0].copy()

        # plot and save orbit
        if self.w.shape[-1] == 6:
            fig = gd.plot_orbits(self.w, marker='.', linestyle='none', alpha=0.2)
            fig.savefig(os.path.join(plot_path,"orbit_{}.png".format(self.name)))

            # plot energy conservation
            E = self.potential.total_energy(self.w[:,0,:3],self.w[:,0,3:])
            plt.clf()
            plt.semilogy(self.t[1:], np.abs(E[1:]-E[:-1]), marker=None)
            plt.savefig(os.path.join(plot_path,"energy_{}.png".format(self.name)))

    def test_naff(self):
        naff = NAFF(self.t)

        ndim = len(self.true_freqs)
        w = np.zeros((len(self.w),2*ndim))
        for i in range(ndim):
            w[:,i] = self.w[:,0,i]
            w[:,i+ndim] = self.w[:,0,i+3]
        f,d,ixes = naff.find_fundamental_frequencies(w, nvec=15)

        logger.info("True freqs: {}".format(self.true_freqs))
        logger.info("Find freqs: {}".format(f))

        done = []
        for freq in f:
            for i,tfreq in enumerate(self.true_freqs):
                if i not in done:
                    if abs(abs(tfreq) - abs(freq)) < 1E-3:
                        done.append(i)
        assert len(done) == len(self.true_freqs)

# -----------------------------------------------------------------------
# Hand-constructed time-series
#
class TestHandConstructed(NAFFBase):
    def setup(self):
        Ts = np.array([1., 1.2, 1.105])
        As = (1., 0.5, 0.2)
        self.t = np.linspace(0,100,50000)
        self.true_freqs = (2*np.pi) / Ts

        f = np.sum([A*(np.cos(2*np.pi*self.t/T) + 1j*np.sin(2*np.pi*self.t/T)) for T,A in zip(Ts,As)], axis=0)
        self.w = np.vstack((f.real, f.imag)).T[:,np.newaxis]

        self.dt = 0.1
        self.nsteps = 20000

# -----------------------------------------------------------------------
# Harmonic Oscillator
#
class TestHarmonicOscillator(NAFFBase):
    def setup_class(self):
        Ts = np.array([1., 1.2, 1.105])
        self.true_freqs = 2*np.pi/Ts
        self.potential = gp.HarmonicOscillatorPotential(self.true_freqs)
        self.w0 = np.array([1,0,0.2,0.,0.1,-0.8])

        self.dt = 0.009
        self.nsteps = 20000

# -----------------------------------------------------------------------
# Logarithmic potential, 1D orbit as in Papaphilippou & Laskar (1996), Table 2
#
class TestLogarithmic1D(NAFFBase):
    def setup_class(self):
        self.true_freqs = np.array([2.13905125])
        self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                                 q1=1., q2=0.9, q3=1., units=galactic)
        self.w0 = np.array([0.49,0.,0.,1.4,0.,0.])

        self.dt = 0.005
        self.nsteps = 2**15

# -----------------------------------------------------------------------
# Logarithmic potential, 2D orbit as in Papaphilippou & Laskar (1996), Table 1
#
class TestLogarithmic2D(NAFFBase):
    def setup_class(self):
        self.true_freqs = np.array([2.16326132, 3.01405257])
        self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                                 q1=1., q2=0.9, q3=1., units=galactic)
        self.w0 = np.array([0.49,0.,0.,1.3156,0.4788,0.])

        self.dt = 0.005
        self.nsteps = 2**15

# -----------------------------------------------------------------------
# Axisymmetric potential
# TODO: need to compare to Sanders' ?
class TestAxisymmetric(NAFFBase):
    def setup_class(self):
        # self.true_freqs = np.array([2.16326132, 3.01405257, 100.])
        self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                                 q1=1., q2=0.9, q3=1., units=galactic)
        self.w0 = np.array([10.,0,0.,0.,0.15,0.005])

        self.dt = 0.005
        self.nsteps = 2**15
