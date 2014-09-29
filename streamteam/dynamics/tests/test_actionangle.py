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

class TestActions(object):

    def setup(self):
        self.units = (u.kpc, u.Msun, u.Myr)
        self.potential = PW14Potential()
        self.N = 100
        np.random.seed(42)
        w0 = isotropic_w0(N=self.N)
        nsteps = 200000

        if not os.path.exists(os.path.join(this_path, "w.npy")):
            logger.debug("Integrating orbits")
            t,w = self.potential.integrate_orbit(w0, dt=0.2, nsteps=nsteps)

            logger.debug("Saving orbits")
            np.save(os.path.join(this_path, "t.npy"), t)
            np.save(os.path.join(this_path, "w.npy"), w)
        else:
            logger.debug("Loaded orbits")
            t = np.load(os.path.join(this_path, "t.npy"))
            w = np.load(os.path.join(this_path, "w.npy"))

        self.t = t[::10]
        self.w = w[::10]

    def test_classify(self):
        # my classify
        orb_type = classify_orbit(self.w)

        # compare to Sanders'
        for j in range(self.N):
            sdrs = genfunc_3d.assess_angmom(self.w[:,j])
            logger.debug("APW: {}, Sanders: {}".format(orb_type[j], sdrs))
            assert np.all(orb_type[j] == sdrs)

    def test_actions(self, plot=False):
        t = self.t

        N_max = 6
        for n in range(self.N):
            print("\n\n")
            logger.info("======================= Orbit {} =======================".format(n))
            w = self.w[:,n]

            if plot:
                logger.debug("Plotting orbit...")
                fig = plot_orbits(w, marker='.', alpha=0.2, linestyle='none')
                fig.savefig(os.path.join(plot_path,"orbit_{}.png".format(n)))
                plt.close('all')

            logger.debug("Computing actions...")
            actions,angles,freqs = find_actions(t, w, N_max=N_max, usys=self.units)

            # get values from Sanders' code
            logger.debug("Computing actions from genfunc...")
            s_actions,s_angles,s_freqs = sanders_act_ang_freq(t, w, N_max=N_max)
            s_actions = np.abs(s_actions)
            s_freqs = np.abs(s_freqs)

            logger.info("Action ratio: {}".format(actions / s_actions))
            logger.info("Angle ratio: {}".format(angles / s_angles))
            logger.info("Freq ratio: {}".format(freqs / s_freqs))

            assert np.allclose(actions, s_actions, rtol=1E-5)
            assert np.allclose(angles, s_angles, rtol=1E-5)
            assert np.allclose(freqs, s_freqs, rtol=1E-5)

        fig = plot_angles(t,angles,freqs)
        fig.savefig(os.path.join(plot_path,"loop_angles.png"))

        fig = plot_angles(t,s_angles,s_freqs)
        fig.savefig(os.path.join(plot_path,"loop_angles_sanders.png"))

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

class TestLoopAcctions(object):

    def setup(self):
        self.usys = (u.kpc, u.Msun, u.Myr)
        self.potential = LM10Potential()
        acc = lambda t,x: self.potential.acceleration(x)
        self.integrator = LeapfrogIntegrator(acc)
        self.loop_w0 = np.append(([14.69, 1.8, 0.12]*u.kpc).decompose(self.usys).value,
                                 ([15.97, -128.9, 44.68]*u.km/u.s).decompose(self.usys).value)

    def test_subsample(self):
        N_max = 6
        NT = 9*N_max**3 / 4

        nsteps = 20000
        t,w = self.integrator.run(self.loop_w0, dt=0.5, nsteps=nsteps)

        every = nsteps // NT // 2
        t = t[::every]
        w = w[::every]
        logger.debug("w shape: {}".format(w.shape))

        fig = plot_orbits(w,ix=0,marker=None)
        fig.savefig(os.path.join(plot_path,"subsample_loop.png"))

        actions,angles,freqs = find_actions(t, w[:,0], N_max=N_max, usys=self.usys)

        # get values from Sanders' code
        s_actions,s_angles,s_freqs = sanders_act_ang_freq(t, w, N_max=N_max)
        s_actions = np.abs(s_actions)
        s_freqs = np.abs(s_freqs)

        print("Action ratio:", actions / s_actions)
        print("Angle ratio:", angles / s_angles)
        print("Freq ratio:", freqs / s_freqs)

        fig = plot_angles(t, angles, freqs, subsample_factor=len(t))
        fig.savefig(os.path.join(plot_path,"subsample_loop_angles.png"))

        fig = plot_angles(t, s_angles, s_freqs, subsample_factor=len(t))
        fig.savefig(os.path.join(plot_path,"subsample_loop_angles_sanders.png"))

        assert np.allclose(actions, s_actions, rtol=1E-2)
        assert np.allclose(angles, s_angles, rtol=1E-2)
        assert np.allclose(freqs, s_freqs, rtol=1E-2)

    def test_actions(self):
        t,w = self.integrator.run(self.loop_w0, dt=0.5, nsteps=20000)

        fig = plot_orbits(w,ix=0,marker=None)
        fig.savefig(os.path.join(plot_path,"loop.png"))

        N_max = 6
        actions,angles,freqs = find_actions(t, w[:,0], N_max=N_max, usys=self.usys)

        # get values from Sanders' code
        s_actions,s_angles,s_freqs = sanders_act_ang_freq(t, w, N_max=N_max)
        s_actions = np.abs(s_actions)
        s_freqs = np.abs(s_freqs)

        print("Action ratio:", actions / s_actions)
        print("Angle ratio:", angles / s_angles)
        print("Freq ratio:", freqs / s_freqs)

        fig = plot_angles(t,angles,freqs)
        fig.savefig(os.path.join(plot_path,"loop_angles.png"))

        fig = plot_angles(t,s_angles,s_freqs)
        fig.savefig(os.path.join(plot_path,"loop_angles_sanders.png"))

        assert np.allclose(actions, s_actions, rtol=1E-2)
        assert np.allclose(angles, s_angles, rtol=1E-2)
        assert np.allclose(freqs, s_freqs, rtol=1E-2)

    def test_given_toy(self):
        t,w = self.integrator.run(self.loop_w0, dt=0.5, nsteps=20000)

        m,b = fit_isochrone(w, usys=self.usys)
        toy_potential = IsochronePotential(m=m, b=b, usys=self.usys)

        N_max = 6
        actions,angles,freqs = find_actions(t, w[:,0], N_max=N_max, usys=self.usys,
                                            toy_potential=toy_potential)

    def test_cross_validate(self):
        N_max = 6

        # integrate a long orbit
        logger.debug("Integrating orbit...")
        t,w = self.integrator.run(self.loop_w0, dt=0.5, nsteps=200000)
        logger.debug("Orbit integration done")

        actions,angles,freqs = cross_validate_actions(t, w[:,0], N_max=N_max, usys=self.usys)
        action_std = (np.std(actions, axis=0)*u.kpc**2/u.Myr).to(u.kpc*u.km/u.s)
        freq_std = (np.std(freqs, axis=0)/u.Myr).to(1/u.Gyr)

        # Sanders' reported variance is ∆J = (0.07, 0.08, 0.03)
        #                               ∆Ω = (3e-4, 6e-5, 2e-3)
        print(action_std)
        print(freq_std)

class TestDifficultAcctions(object):

    def setup(self):
        path = os.path.split(os.path.abspath(__file__))[0]
        self.usys = (u.kpc, u.Msun, u.Myr)
        params = {'a': 6.5, 'q1': 1.3, 'c': 0.3, 'b': 0.26, 'q3': 0.8, 'r_h': 30.0,
                  'm_disk': 65000000000.0, 'psi': 1.570796, 'q2': 1.0, 'theta': 1.570796,
                  'phi': 1.570796, 'm_spher': 20000000000.0, 'v_h': 0.5600371815834104}
        self.potential = PW14Potential(**params)
        acc = lambda t,w: np.hstack((w[...,3:],self.potential.acceleration(w[...,:3])))
        self.integrator = DOPRI853Integrator(acc)
        # self.w0 = np.append(([20., 2.5, 0.]*u.kpc).decompose(self.usys).value,
        #                     ([0., 0., 146.66883]*u.km/u.s).decompose(self.usys).value)

        if not os.path.exists(os.path.join(path, "w.npy")):
            self.w0 = np.loadtxt(os.path.join(path, "w0.txt"))
            logger.debug("Integrating orbits")
            t,w = self.integrator.run(self.w0, dt=0.2, nsteps=50000)

            logger.debug("Saving orbits")
            np.save(os.path.join(path, "t.npy"), t)
            np.save(os.path.join(path, "w.npy"), w)
        else:
            logger.debug("Loaded orbits")
            t = np.load(os.path.join(path, "t.npy"))
            w = np.load(os.path.join(path, "w.npy"))

        self.t = t[::10]
        self.w = w[::10,0][:,np.newaxis]

    def test_toy_potentials(self):
        import toy_potentials  # sanders

        toy_potential = fit_toy_potential(self.w, self.usys)
        actions,angles = toy_potential.action_angle(self.w[:,0,:3], self.w[:,0,3:])

        params = (toy_potential.parameters['m']/1E11, toy_potential.parameters['b'])
        s_w = self.w[:,0].copy()
        s_w[:,3:] = (s_w[:,3:]*u.kpc/u.Myr).to(u.km/u.s).value
        AA = np.array([toy_potentials.angact_iso(i,params) for i in s_w])
        AA = AA[~np.isnan(AA).any(1)]
        s_actions = (AA[:,:3]*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr).value
        s_angles = AA[:,3:]

        assert np.all(np.abs(np.sum(actions - s_actions, axis=0) / len(actions)) < 1E-13)
        assert np.all(np.abs(np.sum(angles - s_angles, axis=0) / len(angles)) < 1E-13)

    def test_actions(self):
        t = self.t
        w = self.w

        fig = plot_orbits(w,ix=0,marker=None)
        fig.savefig(os.path.join(plot_path,"difficult_orbit.png"))

        N_max = 6
        actions,angles,freqs = find_actions(t, w[:,0], N_max=N_max, usys=self.usys)

        # get values from Sanders' code
        s_actions,s_angles,s_freqs = sanders_act_ang_freq(t, w, N_max=N_max)
        s_actions = np.abs(s_actions)
        s_freqs = np.abs(s_freqs)

        print("actions: ", actions)
        print("angles: ", angles)
        print("freqs: ", freqs)
        print()
        print("s actions: ", s_actions)
        print("s angles: ", s_angles)
        print("s freqs: ", s_freqs)
        print()
        print("Action ratio:", actions / s_actions)
        print("Angle ratio:", angles / s_angles)
        print("Freq ratio:", freqs / s_freqs)

        return

        fig = plot_angles(t,angles,freqs)
        fig.savefig(os.path.join(plot_path,"difficult_angles.png"))

        fig = plot_angles(t,s_angles,s_freqs)
        fig.savefig(os.path.join(plot_path,"difficult_angles_sanders.png"))

        assert np.allclose(actions, s_actions, rtol=1E-2)
        assert np.allclose(angles, s_angles, rtol=1E-2)
        assert np.allclose(freqs, s_freqs, rtol=1E-2)

# class TestFrequencyMap(object):

#     def setup(self):
#         self.usys = (u.kpc, u.Msun, u.Myr)
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
#                                                             usys=self.usys, skip_failures=True)
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
