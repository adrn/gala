# coding: utf-8

""" Test action-angle stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import logging

# Third-party
from astropy import log as logger
from astropy.utils.console import color_print
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

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics/naff"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")


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

class TestNAFF(object):

    def setup(self):
        name = self.__class__.__name__
        logger.debug("-"*79)
        logger.debug(name)

        # plot and save orbit
        if self.w.shape[-1] == 6:
            fig = gd.plot_orbits(self.w, marker='.', linestyle='none', alpha=0.2)
            fig.savefig(os.path.join(plot_path,"orbit_{}.png".format(name)))

            # plot energy conservation
            E = self.potential.total_energy(self.w[:,0,:3],self.w[:,0,3:])
            plt.semilogy(self.t[1:], np.abs(E[1:]-E[:-1]), marker=None)

    def test_naff(self):
        naff = NAFF(self.t)
        f,d,ixes = naff.find_fundamental_frequencies(self.w, nvec=15)

# -----------------------------------------------------------------------
# Hand-constructed time-series
#
class TestHandConstructed(TestNAFF):
    def setup(self):
        Ts = np.array([1., 1.2, 1.105])
        As = (1., 0.5, 0.2)
        self.t = np.linspace(0,100,50000)
        self.true_freqs = (2*np.pi) / Ts

        f = np.sum([A*(np.cos(2*np.pi*self.t/T) + 1j*np.sin(2*np.pi*self.t/T)) for T,A in zip(Ts,As)], axis=0)
        self.w = np.vstack((f.real, f.imag)).T

        super(TestHandConstructed,self).setup()

# -----------------------------------------------------------------------
# Harmonic Oscillator
#
class TestHarmonicOscillator(TestNAFF):
    def setup(self):
        Ts = np.array([1., 1.2, 1.105])
        self.true_freqs = 2*np.pi/Ts

        t2 = 100
        nsteps = 50000
        dt = t2/float(nsteps)

        self.potential = gp.HarmonicOscillatorPotential(self.true_freqs)
        self.t,self.w = self.potential.integrate_orbit([1,0,0.2,0.,0.1,-0.8],
                                                       dt=dt, nsteps=nsteps,
                                                       Integrator=DOPRI853Integrator)

        super(TestHarmonicOscillator,self).setup()
