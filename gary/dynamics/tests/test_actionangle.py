# coding: utf-8

""" Test action-angle stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging

# Third-party
import numpy as np
from astropy import log as logger
from scipy.linalg import solve
import pytest

# Project
from ...integrate import DOPRI853Integrator
from ...potential import (IsochronePotential, HarmonicOscillatorPotential,
                          LeeSutoTriaxialNFWPotential)
from ...units import galactic
from ..actionangle import *
from ..core import *
from ..plot import *
from .helpers import *

logger.setLevel(logging.ERROR)

def test_generate_n_vectors():
    # test against Sanders' method
    nvecs = generate_n_vectors(N_max=6, dx=2, dy=2, dz=2)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=2, dy=2, dz=2)
    assert np.all(nvecs == nvecs_sanders)

    nvecs = generate_n_vectors(N_max=6, dx=1, dy=1, dz=1)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=1, dy=1, dz=1)
    assert np.all(nvecs == nvecs_sanders)

def test_fit_isochrone():
    # integrate orbit in Isochrone potential, then try to recover it
    true_m = 2.81E11
    true_b = 11.
    potential = IsochronePotential(m=true_m, b=true_b, units=galactic)
    orbit = potential.integrate_orbit([15.,0,0,0,0.2,0], dt=2., n_steps=10000)

    fit_potential = fit_isochrone(orbit)
    m,b = fit_potential.parameters['m'].value, fit_potential.parameters['b'].value
    assert np.allclose(m, true_m, rtol=1E-2)
    assert np.allclose(b, true_b, rtol=1E-2)

def test_fit_harmonic_oscillator():
    # integrate orbit in harmonic oscillator potential, then try to recover it
    true_omegas = np.array([0.011, 0.032, 0.045])
    potential = HarmonicOscillatorPotential(omega=true_omegas, units=galactic)
    orbit = potential.integrate_orbit([15.,1,2,0,0,0], dt=2., n_steps=10000)

    fit_potential = fit_harmonic_oscillator(orbit)
    omegas = fit_potential.parameters['omega'].value
    assert np.allclose(omegas, true_omegas, rtol=1E-2)

def test_fit_toy_potential():
    # integrate orbit in both toy potentials, make sure correct one is chosen
    true_m = 2.81E11
    true_b = 11.
    true_potential = IsochronePotential(m=true_m, b=true_b, units=galactic)
    orbit = true_potential.integrate_orbit([15.,0,0,0,0.2,0], dt=2., n_steps=10000)

    potential = fit_toy_potential(orbit)
    for k,v in true_potential.parameters.items():
        assert np.allclose(v.value, potential.parameters[k], rtol=1E-2)

    # -----------------------------------------------------------------
    true_omegas = np.array([0.011, 0.032, 0.045])
    true_potential = HarmonicOscillatorPotential(omega=true_omegas, units=galactic)
    orbit = true_potential.integrate_orbit([15.,1,2,0,0,0], dt=2., n_steps=10000)

    potential = fit_toy_potential(orbit)
    assert np.allclose(potential.parameters['omega'].value,
                       true_potential.parameters['omega'].value,
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
        # periods = 2*np.pi/omegas
        # print("Periods:", periods)
        # print("N periods:", t.max() / periods)

        angles = t[np.newaxis] * omegas[:,np.newaxis]
        checks,failures = check_angle_sampling(nvecs, angles)

        assert np.all(failures == i)

class ActionsBase(object):

    @pytest.fixture(autouse=True)
    def set_tmpdir(self, tmpdir):
        self.tmpdir = tmpdir

    def test_classify(self):
        # my classify
        orb_type = self.orbit.circulation()

        # compare to Sanders'
        for j in range(self.N):
            sdrs = genfunc_3d.assess_angmom(self.w[...,j].T)
            logger.debug("APW: {}, Sanders: {}".format(orb_type[:,j], sdrs))
            assert np.all(orb_type[:,j] == sdrs)

    def test_actions(self):
        # t = self.t[::10]
        t = self.t

        N_max = 6
        for n in range(self.N):
            print("\n\n")
            logger.info("======================= Orbit {} =======================".format(n))
            # w = self.w[:,::10,n]
            w = self.w[...,n]
            orb = self.orbit[:,n]
            circ = orb.circulation()

            # get values from Sanders' code
            logger.debug("Computing actions from genfunc...")
            s_actions,s_angles,s_freqs,toy_potential = sanders_act_ang_freq(t, w, circ, N_max=N_max)

            logger.debug("Computing actions...")
            ret = find_actions(orb, N_max=N_max, toy_potential=toy_potential)
            actions = ret['actions']
            angles = ret['angles']
            freqs = ret['freqs']

            logger.info("Action ratio: {}".format(actions / s_actions))
            logger.info("Angle ratio: {}".format(angles / s_angles))
            logger.info("Freq ratio: {}".format(freqs / s_freqs))

            assert np.allclose(actions.value, s_actions, rtol=1E-5)
            assert np.allclose(angles.value, s_angles, rtol=1E-5)
            assert np.allclose(freqs.value, s_freqs, rtol=1E-5)

            # logger.debug("Plotting orbit...")
            # fig = plot_orbits(w, marker='.', alpha=0.2, linestyle='none')
            # fig.savefig(str(self.plot_path.join("orbit_{}.png".format(n))))

            # fig = plot_angles(t,angles,freqs)
            # fig.savefig(str(self.plot_path.join("angles_{}.png".format(n))))

            # fig = plot_angles(t,s_angles,s_freqs)
            # fig.savefig(str(self.plot_path.join("angles_sanders_{}.png".format(n))))

            # plt.close('all')

            # print("Plots saved at:", self.plot_path)

class TestActions(ActionsBase):

    def setup(self):
        self.plot_path = self.tmpdir.mkdir("normal")

        self.units = galactic
        self.potential = LeeSutoTriaxialNFWPotential(v_c=0.2, r_s=20.,
                                                     a=1., b=0.77, c=0.55,
                                                     units=galactic)
        self.N = 8
        np.random.seed(42)
        w0 = isotropic_w0(N=self.N)
        n_steps = 20000

        # integrate orbits
        orbit = self.potential.integrate_orbit(w0, dt=2., n_steps=n_steps,
                                               Integrator=DOPRI853Integrator)
        self.orbit = orbit
        self.t = orbit.t.value
        self.w = orbit.w()

# TODO: need to fix this -- or assess whether needed?
# class TestHardActions(ActionsBase):

#     def setup(self):
#         self.plot_path = self.tmpdir.mkdir("hard")

#         self.units = (u.kpc, u.Msun, u.Myr)
#         params = {'disk': {'a': 6.5, 'b': 0.26, 'm': 65000000000.0},
#                   'bulge': {'c': 0.3, 'm': 20000000000.0},
#                   'halo': {'psi': 1.570796, 'theta': 1.570796, 'phi': 1.570796,
#                            'a': 1., 'b': 0.77, 'c': 0.61, 'r_s': 30.0, 'v_c': 0.22}}
#         self.potential = PW14Potential(**params)
#         self.N = 25

#         w0 = np.loadtxt(os.path.join(test_data_path, "w0.txt"))
#         n_steps = 200000

#         if not os.path.exists(os.path.join(test_data_path, "w_hard.npy")):
#             logger.debug("Integrating orbits")
#             t,w = self.potential.integrate_orbit(w0, dt=0.2, n_steps=n_steps, Integrator=DOPRI853Integrator)

#             logger.debug("Saving orbits")
#             np.save(os.path.join(test_data_path, "t_hard.npy"), t)
#             np.save(os.path.join(test_data_path, "w_hard.npy"), w)
#         else:
#             logger.debug("Loaded orbits")
#             t = np.load(os.path.join(test_data_path, "t_hard.npy"))
#             w = np.load(os.path.join(test_data_path, "w_hard.npy"))

#         self.t = t
#         self.w = w

def test_compare_action_prepare():

    from ..actionangle import _action_prepare, _angle_prepare

    logger.setLevel(logging.ERROR)
    AA = np.random.uniform(0., 100., size=(1000,6))
    t = np.linspace(0., 100., 1000)

    act_san,n_vectors = solver.solver(AA, N_max=6, symNx=2)
    A2,b2,n = _action_prepare(AA.T, N_max=6, dx=2, dy=2, dz=2)
    act_apw = np.array(solve(A2,b2))

    ang_san = solver.angle_solver(AA, t, N_max=6, symNx=2, sign=1)
    A2,b2,n = _angle_prepare(AA.T, t, N_max=6, dx=2, dy=2, dz=2)
    ang_apw = np.array(solve(A2,b2))

    assert np.allclose(act_apw, act_san)
    # assert np.allclose(ang_apw, ang_san)

    # TODO: this could be critical -- why don't our angles agree?
