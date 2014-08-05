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

# Project
from ...integrate import LeapfrogIntegrator
from ...potential import LogarithmicPotential
from ...potential import NFWPotential
from ...potential.lm10 import LM10Potential
from ..actionangle import *
from ..core import *
from ..plot import *

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics/actionangle"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# HACK:
sys.path.append("/Users/adrian/Downloads/genfunc-master")
import genfunc_3d

def angmom(x):
    return np.array([x[1]*x[5]-x[2]*x[4],x[2]*x[3]-x[0]*x[5],x[0]*x[4]-x[1]*x[3]])

def sanders_classify(X):
    L=angmom(X[0])
    loop = np.array([1,1,1])
    for i in X[1:]:
        L0 = angmom(i)
        if(L0[0]*L[0]<0.):
            loop[0] = 0
        if(L0[1]*L[1]<0.):
            loop[1] = 0
        if(L0[2]*L[2]<0.):
            loop[2] = 0
    return loop

def test_classify():
    usys = (u.kpc, u.Msun, u.Myr)
    potential = NFWPotential(v_h=(121.858*u.km/u.s).decompose(usys).value,
                             r_h=20., q1=0.86, q2=1., q3=1.18, usys=usys)
    acc = lambda t,x: potential.acceleration(x)
    integrator = LeapfrogIntegrator(acc)

    # initial conditions
    loop_w0 = [[6.975016793191392, -93.85342183505938, -71.90978460109265, -0.19151220547102255, -0.5944685489722188, 0.4262481187389783], [-119.85377948180077, -50.68671610744867, -10.05148560039928, -0.3351091185863992, -0.42681239582943836, -0.2512200315205476]]
    t,loop_ws = integrator.run(loop_w0, dt=1., nsteps=15000)

    box_w0 = [[57.66865614916953, -66.09241133078703, 47.43779192106421, -0.6862780950091272, 0.04550073987392385, -0.36216991360120393], [-12.10727872905934, -17.556470673741607, 7.7552881580976, -0.1300187288715955, -0.023618199542192752, 0.08686283408067244]]
    t,box_ws = integrator.run(box_w0, dt=1., nsteps=15000)

    # my classify
    orb_type = classify_orbit(loop_ws)
    for j in range(len(loop_w0)):
        assert np.all(orb_type[j] == sanders_classify(loop_ws[:,j]))

    orb_type = classify_orbit(box_ws)
    for j in range(len(box_w0)):
        assert np.all(orb_type[j] == sanders_classify(box_ws[:,j]))

def _crazy_loop(theta1,theta2,ax):
    cnt = 0
    ix1 = 0
    while True:
        cnt += 1

        for ix2 in range(ix1,ix1+1000):
            if ix2 > len(theta1)-1:
                ix2 = len(theta1)-1
                break

            if theta1[ix2] < theta1[ix1] or theta2[ix2] < theta2[ix1]:
                ix2 -= 1
                break

        if theta1[ix2] != theta1[ix1:ix2+1].max() or theta2[ix2] != theta2[ix1:ix2+1].max():
            ix1 = ix2+1
            continue

        if cnt > 100 or ix2 == len(theta1)-1:
            break

        if ix1 == ix2:
            ix1 = ix2+1
            continue

        ax.plot(theta1[ix1:ix2+1], theta2[ix1:ix2+1], alpha=0.5, marker='o', c='k')

        ix1 = ix2+1

def plot_angles(t,angles,freqs):
    theta = (angles[:,None] + freqs[:,None]*t[np.newaxis])
    subsample = theta.shape[1]//1000
#    subsample = 1
    theta = (theta[:,::subsample] / np.pi) % 2.

    fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
    _crazy_loop(theta[0], theta[1], axes[0])
    _crazy_loop(theta[0], theta[2], axes[1])

    axes[0].set_xlim(0,2)
    axes[0].set_ylim(0,2)
    return fig
    # axes[1].scatter(theta[0,ix], theta[2], alpha=0.5, marker='o', c=t)

def sanders_nvecs(N_max, dx, dy, dz):
    from itertools import product
    NNx = range(-N_max, N_max+1, dx)
    NNy = range(-N_max, N_max+1, dy)
    NNz = range(-N_max, N_max+1, dz)
    n_vectors = np.array([[i,j,k] for (i,j,k) in product(NNx,NNy,NNz)
                          if(not(i==0 and j==0 and k==0)            # exclude zero vector
                             and (k>0                               # northern hemisphere
                                  or (k==0 and j>0)                 # half of x-y plane
                                  or (k==0 and j==0 and i>0))       # half of x axis
                             and np.sqrt(i*i+j*j+k*k)<=N_max)])     # inside sphere
    return n_vectors

def test_nvecs():
    nvecs = generate_n_vectors(N_max=6, dx=2, dy=2, dz=2)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=2, dy=2, dz=2)

    assert np.all(nvecs == nvecs_sanders)

# def test_compare_action_prepare():
#     from ..actionangle import _action_prepare
#     import solver
#     logger.setLevel(logging.ERROR)
#     AA = np.random.uniform(0., 100., size=(1000,6))

#     A1,b1 = solver.solver(AA, N_max=6, symNx=2)
#     A2,b2,n = _action_prepare(AA, N_max=6, dx=2, dy=2, dz=2)

#     assert np.allclose(A1, A2)
#     assert np.allclose(b1, b2)

# def test_compare_angle_prepare():
#     from ..actionangle import _angle_prepare
#     import solver
#     logger.setLevel(logging.ERROR)
#     AA = np.random.uniform(0., 100., size=(1000,6))
#     t = np.linspace(0., 100., 1000)

#     A1,b1 = solver.angle_solver(AA, t, N_max=6, sign=1., symNx=2)
#     A2,b2,n = _angle_prepare(AA, t, N_max=6, dx=2, dy=2, dz=2)

#     # row = slice(None,None)
#     # col = slice(None,None)
#     # assert np.allclose(A1[row,col], A2[row,col])

#     assert np.allclose(A1, A2)
#     assert np.allclose(b1, b2)

def sanders_act_ang_freq(t,w,N_max=6):
    w2 = w.copy()
    w2[:,0,3:] = (w2[:,0,3:]*u.kpc/u.Myr).to(u.km/u.s).value
    act,ang,n_vec,toy_aa,pars = genfunc_3d.find_actions(w2[:,0], t/1000., N_matrix=N_max)

    actions = (act[:3]*u.kpc*u.km/u.s).to(u.kpc**2/u.Myr).value
    angles = ang[:3]
    freqs = (ang[3:6]/u.Gyr).to(1/u.Myr).value

    return actions,angles,freqs

class TestLoopActions(object):

    def setup(self):
        self.usys = (u.kpc, u.Msun, u.Myr)
        self.potential = LM10Potential()
        acc = lambda t,x: self.potential.acceleration(x)
        self.integrator = LeapfrogIntegrator(acc)
        self.loop_w0 = np.append(([14.69, 1.8, 0.12]*u.kpc).decompose(self.usys).value,
                                 ([15.97, -128.9, 44.68]*u.km/u.s).decompose(self.usys).value)

    def test_actions(self):
        t,w = self.integrator.run(self.loop_w0, dt=0.5, nsteps=20000)

        fig = plot_orbit(w,ix=0,marker=None)
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

class TestFrequencyMap(object):

    def setup(self):
        self.usys = (u.kpc, u.Msun, u.Myr)
        self.potential = LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
                                              q1=1., q2=1., q3=0.7, phi=0.)
        acc = lambda t,x: self.potential.acceleration(x)
        self.integrator = LeapfrogIntegrator(acc)

        n = 3
        phis = np.linspace(0.1,1.95*np.pi,n)
        thetas = np.arccos(2*np.linspace(0.05,0.95,n) - 1)
        p,t = np.meshgrid(phis, thetas)
        phis = p.ravel()
        thetas = t.ravel()

        sinp,cosp = np.sin(phis),np.cos(phis)
        sint,cost = np.sin(thetas),np.cos(thetas)

        rh2 = self.potential.parameters['r_h']**2
        q2 = self.potential.parameters['q2']
        q3 = self.potential.parameters['q3']
        r2 = (np.e - rh2) / (sint**2*cosp**2 + sint**2*sinp**2/q2**2 + cost**2/q3**2)
        r = np.sqrt(r2)

        x = r*cosp*sint
        y = r*sinp*sint
        z = r*cost
        v = np.zeros_like(x)

        E = self.potential.energy(np.vstack((x,y,z)).T, np.vstack((v,v,v)).T)
        assert np.allclose(E, 0.5)

        self.grid = np.vstack((x,y,z,v,v,v)).T

    def test(self):
        N_max = 6
        logger.debug("Integrating orbits...")
        t,w = self.integrator.run(self.grid, dt=0.05, nsteps=100000)
        logger.debug("...done!")

        fig,axes = plt.subplots(1,3,figsize=(16,5))
        all_freqs = []
        for n in range(w.shape[1]):
            try:
                actions,angles,freqs = cross_validate_actions(t, w[:,n], N_max=N_max,
                                                            usys=self.usys, skip_failures=True)
                failed = False
            except ValueError as e:
                print("FAILED: {}".format(e))
                failed = True

            if not failed:
                all_freqs.append(freqs)

            fig = plot_orbit(w, ix=n, axes=axes, linestyle='none', marker='.', alpha=0.1)
            fig.axes[1].set_title("Failed: {}".format(failed),fontsize=24)
            fig.savefig(os.path.join(plot_path,"orbit_{}.png".format(n)))
            for i in range(3): axes[i].cla()

        # for freqs in all_freqs:
        #     print(np.median(freqs,axis=0))
        #     print(np.std(freqs,axis=0))

        all_freqs = np.array(all_freqs)

        plt.clf()
        plt.figure(figsize=(6,6))
        plt.plot(all_freqs[:,1]/all_freqs[:,0], all_freqs[:,2]/all_freqs[:,0],
                 linestyle='none', marker='.')
        # plt.xlim(0.9, 1.5)
        # plt.ylim(1.1, 2.1)
        plt.savefig(os.path.join(plot_path,"freq_map.png"))
