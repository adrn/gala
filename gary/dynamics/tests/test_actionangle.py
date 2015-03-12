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

# Project
from ...integrate import DOPRI853Integrator
from ...potential import IsochronePotential, HarmonicOscillatorPotential
from ...potential import LeeSutoTriaxialNFWPotential
from ...potential.custom import PW14Potential
from ...units import galactic
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

# TODO: config item to specify path to test data?
test_data_path = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                 "../../../test-data/actionangle"))

def test_generate_n_vectors():
    # test against Sanders' method
    nvecs = generate_n_vectors(N_max=6, dx=2, dy=2, dz=2)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=2, dy=2, dz=2)
    assert np.all(nvecs == nvecs_sanders)

    nvecs = generate_n_vectors(N_max=6, dx=1, dy=1, dz=1)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=1, dy=1, dz=1)
    assert np.all(nvecs == nvecs_sanders)

def test_unwrap_angles():
    # generate fake angles
    t = np.linspace(10,100.,250)
    omegas = np.array([0.21, 0.32, 0.55])
    angles = t[:,np.newaxis] * omegas[np.newaxis]
    wrap_angles = angles % (2*np.pi)

    unwrapped_angles = unwrap_angles(wrap_angles, sign=1.)
    assert np.allclose(angles, unwrapped_angles)

def test_fit_isochrone():
    # integrate orbit in Isochrone potential, then try to recover it
    true_m = 2.81E11
    true_b = 11.
    potential = IsochronePotential(m=true_m, b=true_b, units=galactic)
    t,w = potential.integrate_orbit([15.,0,0,0,0.2,0], dt=2., nsteps=10000)

    m,b = fit_isochrone(w, units=galactic)
    assert np.allclose(m, true_m, rtol=1E-2)
    assert np.allclose(b, true_b, rtol=1E-2)

def test_fit_harmonic_oscillator():
    # integrate orbit in harmonic oscillator potential, then try to recover it
    true_omegas = np.array([0.011, 0.032, 0.045])
    potential = HarmonicOscillatorPotential(omega=true_omegas, units=galactic)
    t,w = potential.integrate_orbit([15.,1,2,0,0,0], dt=2., nsteps=10000)

    omegas = fit_harmonic_oscillator(w, units=galactic)
    assert np.allclose(omegas, true_omegas, rtol=1E-2)

def test_fit_toy_potential():
    # integrate orbit in both toy potentials, make sure correct one is chosen
    true_m = 2.81E11
    true_b = 11.
    true_potential = IsochronePotential(m=true_m, b=true_b, units=galactic)
    t,w = true_potential.integrate_orbit([15.,0,0,0,0.2,0], dt=2., nsteps=10000)

    potential = fit_toy_potential(w, units=galactic)
    for k,v in true_potential.parameters.items():
        assert np.allclose(v, potential.parameters[k], rtol=1E-2)

    # -----------------------------------------------------------------
    true_omegas = np.array([0.011, 0.032, 0.045])
    true_potential = HarmonicOscillatorPotential(omega=true_omegas, units=galactic)
    t,w = true_potential.integrate_orbit([15.,1,2,0,0,0], dt=2., nsteps=10000)

    potential = fit_toy_potential(w, units=galactic)
    assert np.allclose(potential.parameters['omega'],
                       true_potential.parameters['omega'],
                       rtol=1E-2)

def test_check_angle_sampling():

    # frequencies
    omegas = np.array([0.21, 0.3421, 0.4968])

    # integer vectors
    nvecs = generate_n_vectors(N_max=6)

    # loop over times with known failures:
    #   - first one fails needing longer integration time
    #   - second one fails needing finer sampling
    for i,t in enumerate([np.linspace(0,50,500), np.linspace(0,8000,8000)]):
        angles = t[:,np.newaxis] * omegas[np.newaxis]
        periods = 2*np.pi/omegas
        print("Periods:", periods)
        print("N periods:", t.max() / periods)

        angles = t[:,np.newaxis] * omegas[np.newaxis]
        checks,failures = check_angle_sampling(nvecs, angles)

        assert np.all(failures == i)

class ActionsBase(object):

    def test_classify(self):
        # my classify
        orb_type = classify_orbit(self.w)

        # compare to Sanders'
        for j in range(self.N):
            sdrs = genfunc_3d.assess_angmom(self.w[:,j])
            logger.debug("APW: {}, Sanders: {}".format(orb_type[j], sdrs))
            assert np.all(orb_type[j] == sdrs)

    def test_actions(self):
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
        self.potential = LeeSutoTriaxialNFWPotential(v_c=0.2, r_s=20.,
                                                     a=1., b=0.77, c=0.55,
                                                     units=galactic)
        self.N = 25
        np.random.seed(42)
        w0 = isotropic_w0(N=self.N)
        nsteps = 200000

        if not os.path.exists(os.path.join(test_data_path, "w.npy")):
            logger.debug("Integrating orbits")
            t,w = self.potential.integrate_orbit(w0, dt=0.2, nsteps=nsteps, Integrator=DOPRI853Integrator)

            logger.debug("Saving orbits")
            np.save(os.path.join(test_data_path, "t.npy"), t)
            np.save(os.path.join(test_data_path, "w.npy"), w)
        else:
            logger.debug("Loaded orbits")
            t = np.load(os.path.join(test_data_path, "t.npy"))
            w = np.load(os.path.join(test_data_path, "w.npy"))

        self.t = t
        self.w = w

class TestHardActions(ActionsBase):

    def setup(self):
        self.plot_path = os.path.join(plot_path, 'hard')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.units = (u.kpc, u.Msun, u.Myr)
        params = {'disk': {'a': 6.5, 'b': 0.26, 'm': 65000000000.0},
                  'bulge': {'c': 0.3, 'm': 20000000000.0},
                  'halo': {'psi': 1.570796, 'theta': 1.570796, 'phi': 1.570796,
                           'a': 1., 'b': 0.77, 'c': 0.61, 'r_s': 30.0, 'v_c': 0.22}}
        self.potential = PW14Potential(**params)
        self.N = 25

        w0 = np.loadtxt(os.path.join(test_data_path, "w0.txt"))
        nsteps = 200000

        if not os.path.exists(os.path.join(test_data_path, "w_hard.npy")):
            logger.debug("Integrating orbits")
            t,w = self.potential.integrate_orbit(w0, dt=0.2, nsteps=nsteps, Integrator=DOPRI853Integrator)

            logger.debug("Saving orbits")
            np.save(os.path.join(test_data_path, "t_hard.npy"), t)
            np.save(os.path.join(test_data_path, "w_hard.npy"), w)
        else:
            logger.debug("Loaded orbits")
            t = np.load(os.path.join(test_data_path, "t_hard.npy"))
            w = np.load(os.path.join(test_data_path, "w_hard.npy"))

        self.t = t
        self.w = w

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
