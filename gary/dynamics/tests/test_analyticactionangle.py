# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import matplotlib.pyplot as pl
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from ..analyticactionangle import *
from ...potential import IsochronePotential, HarmonicOscillatorPotential
from ...units import galactic
from ...util import assert_angles_allclose
from .helpers import *

class TestIsochrone(object):

    def setup(self):
        logger.info("======== Isochrone ========")
        self.N = 100
        np.random.seed(42)
        x = np.random.uniform(-10., 10., size=(3,self.N))
        v = np.random.uniform(-1., 1., size=(3,self.N)) / 33.
        w0 = np.vstack((x,v))

        self.potential = IsochronePotential(units=galactic, m=1.E11, b=5.)
        self.w = self.potential.integrate_orbit(w0, dt=0.1, n_steps=10000)
        self.w = self.w[::10]

    def test(self):
        for n in range(self.N):
            logger.debug("Orbit {}".format(n))

            actions,angles,freqs = isochrone_to_aa(self.w[:,n], self.potential)
            actions = actions.value
            angles = angles.value

            for i in range(3):
                assert np.allclose(actions[i,1:], actions[i,0], rtol=1E-5)

            # Compare to genfunc
            x = self.w.pos.value[...,n]
            v = self.w.vel.value[...,n]
            s_v = (v*u.kpc/u.Myr).to(u.km/u.s).value
            s_w = np.vstack((x,s_v))
            m = self.potential.parameters['m'].value / 1E11
            b = self.potential.parameters['b'].value
            aa = np.array([toy_potentials.angact_iso(s_w[:,i].T, params=(m,b)) for i in range(s_w.shape[1])])
            s_actions = (aa[:,:3]*u.km/u.s*u.kpc).decompose(galactic).value
            s_angles = aa[:,3:]

            assert np.allclose(actions, s_actions.T, rtol=1E-8)
            assert_angles_allclose(angles, s_angles.T, rtol=1E-8)

            # TODO: when I fix actionangle -> xv function, re-add this
            # x2,v2 = isochrone_aa_to_xv(actions, angles, self.potential)

            # assert np.allclose(x, x2, rtol=1E-8)
            # assert np.allclose(v, v2, rtol=1E-8)

class TestHarmonicOscillator(object):

    def setup(self):
        logger.info("======== Harmonic Oscillator ========")
        self.N = 100
        np.random.seed(42)
        x = np.random.uniform(-10., 10., size=(3,self.N))
        v = np.random.uniform(-1., 1., size=(3,self.N)) / 33.
        w0 = np.vstack((x,v))

        self.potential = HarmonicOscillatorPotential(omega=np.array([0.013, 0.02, 0.005]), units=galactic)
        self.w = self.potential.integrate_orbit(w0, dt=0.1, n_steps=10000)
        self.w = self.w[::10]

    def test(self):
        """
            !!!!! NOTE !!!!!
            For Harmonic Oscillator, Sanders' code works for the units I use...
        """
        for n in range(self.N):
            logger.debug("Orbit {}".format(n))

            actions,angles,freq = harmonic_oscillator_to_aa(self.w[:,n], self.potential)
            actions = actions.value
            angles = angles.value

            for i in range(3):
                assert np.allclose(actions[i,1:], actions[i,0], rtol=1E-5)

            # Compare to genfunc
            x = self.w.pos.value[...,n]
            v = self.w.vel.value[...,n]
            s_w = np.vstack((x,v))
            omega = self.potential.parameters['omega'].value
            aa = np.array([toy_potentials.angact_ho(s_w[:,i].T, omega=omega) for i in range(s_w.shape[1])])
            s_actions = aa[:,:3]
            s_angles = aa[:,3:]

            assert np.allclose(actions, s_actions.T, rtol=1E-8)
            assert_angles_allclose(angles, s_angles.T, rtol=1E-8)

            # test roundtrip
            # x2,v2 = harmonic_oscillator_aa_to_xv(actions, angles, self.potential)
            # TODO: transform back??
