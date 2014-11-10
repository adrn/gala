# coding: utf-8

""" Test action-angle stuff """

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
from scipy.linalg import solve
import pytest

# Project
from ...integrate import LeapfrogIntegrator, DOPRI853Integrator
from ...potential import LogarithmicPotential
from ...potential import NFWPotential, IsochronePotential
from ...potential.apw import PW14Potential
from ...potential.lm10 import LM10Potential
from ..actionangle import *
from ..core import *
from ..plot import *
from .helpers import *

# HACK:
if "/Users/adrian/projects/genfunc" not in sys.path:
    sys.path.append("/Users/adrian/projects/genfunc")
import genfunc_3d

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics/actionangle"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

this_path = os.path.split(os.path.abspath(__file__))[0]

class ActionsBase(object):

    def test_classify(self):
        # my classify
        orb_type = classify_orbit(self.w)

        # compare to Sanders'
        for j in range(self.N):
            sdrs = genfunc_3d.assess_angmom(self.w[:,j])
            logger.debug("APW: {}, Sanders: {}".format(orb_type[j], sdrs))
            assert np.all(orb_type[j] == sdrs)

    def test_actions(self, plot=False):
        t = self.t[::10]

        N_max = 6
        for n in range(self.N):
            print("\n\n")
            logger.info("======================= Orbit {} =======================".format(n))
            w = self.w[::10,n]

            # get values from Sanders' code
            logger.debug("Computing actions from genfunc...")
            s_actions,s_angles,s_freqs,toy_potential = sanders_act_ang_freq(t, w, N_max=N_max)

            logger.debug("Computing actions...")
            actions,angles,freqs = find_actions(t, w, N_max=N_max, units=self.units,
                                                toy_potential=toy_potential)

            logger.info("Action ratio: {}".format(actions / s_actions))
            logger.info("Angle ratio: {}".format(angles / s_angles))
            logger.info("Freq ratio: {}".format(freqs / s_freqs))

            assert np.allclose(actions, s_actions, rtol=1E-5)
            assert np.allclose(angles, s_angles, rtol=1E-5)
            assert np.allclose(freqs, s_freqs, rtol=1E-5)

            if plot:
                logger.debug("Plotting orbit...")
                fig = plot_orbits(w, marker='.', alpha=0.2, linestyle='none')
                fig.savefig(os.path.join(self.plot_path,"orbit_{}.png".format(n)))

                fig = plot_angles(t,angles,freqs)
                fig.savefig(os.path.join(self.plot_path,"angles_{}.png".format(n)))

                fig = plot_angles(t,s_angles,s_freqs)
                fig.savefig(os.path.join(self.plot_path,"angles_sanders_{}.png".format(n)))

                plt.close('all')

class TestActions(ActionsBase):

    def setup(self):
        self.plot_path = os.path.join(plot_path, 'normal')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.units = (u.kpc, u.Msun, u.Myr)
        self.potential = PW14Potential()
        self.N = 100
        np.random.seed(42)
        w0 = isotropic_w0(N=self.N)
        nsteps = 200000

        if not os.path.exists(os.path.join(this_path, "w.npy")):
            logger.debug("Integrating orbits")
            t,w = self.potential.integrate_orbit(w0, dt=0.2, nsteps=nsteps, Integrator=DOPRI853Integrator)

            logger.debug("Saving orbits")
            np.save(os.path.join(this_path, "t.npy"), t)
            np.save(os.path.join(this_path, "w.npy"), w)
        else:
            logger.debug("Loaded orbits")
            t = np.load(os.path.join(this_path, "t.npy"))
            w = np.load(os.path.join(this_path, "w.npy"))

        self.t = t
        self.w = w

class TestHardActions(ActionsBase):

    def setup(self):
        self.plot_path = os.path.join(plot_path, 'hard')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.units = (u.kpc, u.Msun, u.Myr)
        params = {'a': 6.5, 'q1': 1.3, 'c': 0.3, 'b': 0.26, 'q3': 0.8, 'r_h': 30.0,
                  'm_disk': 65000000000.0, 'psi': 1.570796, 'q2': 1.0, 'theta': 1.570796,
                  'phi': 1.570796, 'm_spher': 20000000000.0, 'v_h': 0.5600371815834104}
        self.potential = PW14Potential(**params)
        self.N = 100

        w0 = np.loadtxt(os.path.join(this_path, "w0.txt"))
        nsteps = 200000

        if not os.path.exists(os.path.join(this_path, "w_hard.npy")):
            logger.debug("Integrating orbits")
            t,w = self.potential.integrate_orbit(w0, dt=0.2, nsteps=nsteps, Integrator=DOPRI853Integrator)

            logger.debug("Saving orbits")
            np.save(os.path.join(this_path, "t_hard.npy"), t)
            np.save(os.path.join(this_path, "w_hard.npy"), w)
        else:
            logger.debug("Loaded orbits")
            t = np.load(os.path.join(this_path, "t_hard.npy"))
            w = np.load(os.path.join(this_path, "w_hard.npy"))

        self.t = t
        self.w = w

def test_nvecs():
    nvecs = generate_n_vectors(N_max=6, dx=2, dy=2, dz=2)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=2, dy=2, dz=2)
    assert np.all(nvecs == nvecs_sanders)

    nvecs = generate_n_vectors(N_max=6, dx=1, dy=1, dz=1)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=1, dy=1, dz=1)
    assert np.all(nvecs == nvecs_sanders)

def test_compare_action_prepare():

    from ..actionangle import _action_prepare, _angle_prepare
    import solver
    logger.setLevel(logging.ERROR)
    AA = np.random.uniform(0., 100., size=(1000,6))
    t = np.linspace(0., 100., 1000)

    act_san,n_vectors = solver.solver(AA, N_max=6, symNx=2)
    A2,b2,n = _action_prepare(AA, N_max=6, dx=2, dy=2, dz=2)
    act_apw = np.array(solve(A2,b2))

    ang_san = solver.angle_solver(AA, t, N_max=6, symNx=2, sign=1)
    A2,b2,n = _angle_prepare(AA, t, N_max=6, dx=2, dy=2, dz=2)
    ang_apw = np.array(solve(A2,b2))

    assert np.allclose(act_apw, act_san)
    assert np.allclose(ang_apw, ang_san)

# class TestFrequencyMap(object):

#     def setup(self):
#         self.units = (u.kpc, u.Msun, u.Myr)
#         self.potential = LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
#                                               q1=1., q2=1., q3=0.7, phi=0.)
#         acc = lambda t,x: self.potential.acceleration(x)
#         self.integrator = LeapfrogIntegrator(acc)

#         n = 3
#         phis = np.linspace(0.1,1.95*np.pi,n)
#         thetas = np.arccos(2*np.linspace(0.05,0.95,n) - 1)
#         p,t = np.meshgrid(phis, thetas)
#         phis = p.ravel()
#         thetas = t.ravel()

#         sinp,cosp = np.sin(phis),np.cos(phis)
#         sint,cost = np.sin(thetas),np.cos(thetas)

#         rh2 = self.potential.parameters['r_h']**2
#         q2 = self.potential.parameters['q2']
#         q3 = self.potential.parameters['q3']
#         r2 = (np.e - rh2) / (sint**2*cosp**2 + sint**2*sinp**2/q2**2 + cost**2/q3**2)
#         r = np.sqrt(r2)

#         x = r*cosp*sint
#         y = r*sinp*sint
#         z = r*cost
#         v = np.zeros_like(x)

#         E = self.potential.energy(np.vstack((x,y,z)).T, np.vstack((v,v,v)).T)
#         assert np.allclose(E, 0.5)

#         self.grid = np.vstack((x,y,z,v,v,v)).T

#     def test(self):
#         N_max = 6
#         logger.debug("Integrating orbits...")
#         t,w = self.integrator.run(self.grid, dt=0.05, nsteps=100000)
#         logger.debug("...done!")

#         fig,axes = plt.subplots(1,3,figsize=(16,5))
#         all_freqs = []
#         for n in range(w.shape[1]):
#             try:
#                 actions,angles,freqs = cross_validate_actions(t, w[:,n], N_max=N_max,
#                                                             units=self.units, skip_failures=True)
#                 failed = False
#             except ValueError as e:
#                 print("FAILED: {}".format(e))
#                 failed = True

#             if not failed:
#                 all_freqs.append(freqs)

#             fig = plot_orbits(w, ix=n, axes=axes, linestyle='none', marker='.', alpha=0.1)
#             fig.axes[1].set_title("Failed: {}".format(failed),fontsize=24)
#             fig.savefig(os.path.join(plot_path,"orbit_{}.png".format(n)))
#             for i in range(3): axes[i].cla()

#         # for freqs in all_freqs:
#         #     print(np.median(freqs,axis=0))
#         #     print(np.std(freqs,axis=0))

#         all_freqs = np.array(all_freqs)

#         plt.clf()
#         plt.figure(figsize=(6,6))
#         plt.plot(all_freqs[:,1]/all_freqs[:,0], all_freqs[:,2]/all_freqs[:,0],
#                  linestyle='none', marker='.')
#         # plt.xlim(0.9, 1.5)
#         # plt.ylim(1.1, 2.1)
#         plt.savefig(os.path.join(plot_path,"freq_map.png"))
