# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from ..analyticactionangle import *
from ...potential import IsochronePotential, HarmonicOscillatorPotential
from ...units import galactic

# HACK:
if "/Users/adrian/projects/genfunc" not in sys.path:
    sys.path.append("/Users/adrian/projects/genfunc")
import genfunc_3d, toy_potentials

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics/analyticactionangle"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestIsochrone(object):

    def setup(self):
        self.N = 100
        np.random.seed(42)
        x = np.random.uniform(-10., 10., size=(self.N,3))
        v = np.random.uniform(-1., 1., size=(self.N,3)) / 33.
        w0 = np.hstack((x,v))

        self.potential = IsochronePotential(usys=galactic, m=1.E11, b=5.)
        self.t,self.w = self.potential.integrate_orbit(w0, dt=0.1, nsteps=10000)
        self.t = self.t[::10]
        self.w = self.w[::10]

    def test(self):
        for n in range(self.N):
            logger.debug("Orbit {}".format(n))

            x,v = self.w[:,n,:3],self.w[:,n,3:]
            s_v = (v*u.kpc/u.Myr).to(u.km/u.s).value
            s_w = np.hstack((x,s_v))
            actions,angles = isochrone_xv_to_aa(x, v, self.potential)

            for i in range(3):
                assert np.allclose(actions[1:,i], actions[0,i], rtol=1E-5)

            # Compare to genfunc
            m = self.potential.parameters['m'] / 1E11
            b = self.potential.parameters['b']
            aa = np.array([toy_potentials.angact_iso(w, params=(m,b)) for w in s_w])
            s_actions = (aa[:,:3]*u.km/u.s*u.kpc).decompose(galactic).value
            s_angles = aa[:,3:]

            assert np.allclose(actions, s_actions, rtol=1E-8)
            assert np.allclose(angles, s_angles, rtol=1E-8)

            # test roundtrip
            x2,v2 = isochrone_aa_to_xv(actions, angles, self.potential)

            rel_err_x = np.abs((x2 - x) / x)
            rel_err_v = np.abs((v2 - v) / v)

            assert rel_err_x.max() < (1E-8)
            assert rel_err_v.max() < (1E-8)

class TestHarmonicOscillator(object):

    def setup(self):
        self.N = 10
        np.random.seed(42)
        x = np.random.uniform(-10., 10., size=(self.N,3))
        v = np.random.uniform(-1., 1., size=(self.N,3)) / 33.
        w0 = np.hstack((x,v))

        self.potential = HarmonicOscillatorPotential(omega=np.array([0.013, 0.02, 0.005]), usys=galactic)
        self.t,self.w = self.potential.integrate_orbit(w0, dt=0.1, nsteps=10000)

    def test(self):
        for n in range(self.N):
            x,v = self.w[:,n,:3],self.w[:,n,3:]
            actions,angles = harmonic_oscillator_xv_to_aa(x, v, self.potential)

            for i in range(3):
                assert np.allclose(actions[1:,i], actions[0,i], rtol=1E-5)

            # x2,v2 = harmonic_oscillator_aa_to_xv(actions, angles, self.potential)

            # rel_err_x = np.abs((x2 - x) / x)
            # rel_err_v = np.abs((v2 - v) / v)

            # assert rel_err_x.max() < (1E-8)
            # assert rel_err_v.max() < (1E-8)

# def test_roundtrip(self):

#     np.random.seed(4342)
#     n = 10
#     x = np.random.uniform(-10., 10., size=(n,3))
#     v = np.random.uniform(-1., 1., size=(n,3)) / 33.

#     potential = IsochronePotential(usys=self.usys, m=1.E11, b=5.)
#     acc = lambda t,x: potential.acceleration(x)
#     integrator = LeapfrogIntegrator(acc)
#     t,ws = integrator.run(np.hstack((x,v)), dt=1., nsteps=10000)
#     print()

#     for i in range(n):
#         xs = ws[:,i,:3]
#         vs = ws[:,i,3:]

#         # r,phi,theta,vr,vphi,vtheta = cartesian_to_spherical(xs,vs).T
#         # print("True r", r)
#         # print("True φ", phi)
#         # print("True θ", theta)
#         # print("True vr", vr)
#         # print("True vφ", vphi)
#         # print("True vθ", vtheta)

#         actions,angles = potential.action_angle(xs, vs)
#         x,v = potential.phase_space(actions, angles)

#         assert np.allclose(x, xs, rtol=1E-8)
#         assert np.allclose(v, vs, rtol=1E-8)
