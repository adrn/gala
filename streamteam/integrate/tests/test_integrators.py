# coding: utf-8
"""
    Test the integrators.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

# Project
from ..leapfrog import LeapfrogIntegrator
from ..rk5 import RK5Integrator

top_path = "/tmp/streamteam"
plot_path = os.path.join(top_path, "tests/integrate")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def plot(ts, q, p):
    """ Make some helpful plots for testing the integrators. """

    qp = np.squeeze(np.vstack((q.T,p.T)))
    print(qp.shape)

    ndim = qp.shape[0]
    fig,axes = plt.subplots(ndim, ndim, figsize=(4*ndim,4*ndim),
                            sharex='col', sharey='row')

    kwargs = dict(marker='.', linestyle='-')
    for ii in range(ndim):
        for jj in range(ndim):
            axes[jj,ii].plot(qp[ii], qp[jj], linestyle='-',
                             marker='.', alpha=0.75)

    fig.tight_layout()

    for ii in range(ndim):
        for jj in range(ndim):
            if ii > jj:
                axes[jj,ii].set_visible(False)
                continue
    return fig

@pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ])
def test_forward(name, Integrator):
    T = 10.
    acceleration = lambda q: -(2*np.pi/T)**2*q

    dt = 0.1
    t1,t2 = 0, 2.5
    integrator = Integrator(acceleration)
    ts, qs, ps = integrator.run(q_i=[0.], p_i=[1.],
                                t1=t1, t2=t2, dt=dt)

    fig = plot(ts, qs, ps)
    fig.savefig(os.path.join(plot_path,"forward_{0}.png".format(name)))

@pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ])
def test_backward(name, Integrator):
    T = 10.
    acceleration = lambda q: -(2*np.pi/T)**2*q

    dt = -0.1
    t1,t2 = 2.5, 0
    integrator = Integrator(acceleration)
    ts, qs, ps = integrator.run(q_i=[0.], p_i=[1.],
                                t1=t1, t2=t2, dt=dt)

    fig = plot(ts, qs, ps)
    fig.savefig(os.path.join(plot_path,"backward_{0}.png".format(name)))

@pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ])
def test_harmonic_oscillator(name, Integrator):
    T = 10.
    acceleration = lambda q: -(2*np.pi/T)**2*q

    dt = 0.1
    integrator = Integrator(acceleration)
    ts, qs, ps = integrator.run(q_i=[1.], p_i=[0.],
                                dt=dt, nsteps=100)

    fig = plot(ts, qs, ps)
    fig.savefig(os.path.join(plot_path,"harmonic_osc_{0}.png".format(name)))

@pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ])
def test_point_mass(name, Integrator):
    GM = (G * (1.*u.M_sun)).decompose([u.au,u.M_sun,u.year,u.radian]).value

    def acceleration(q):
        a = -GM/(q[:,0]**2+q[:,1]**2)**1.5
        return np.array([q[:,0]*a, q[:,1]*a]).T.copy()

    q_i = np.array([1.0, 0.0]) # au
    p_i = np.array([0.0, 2*np.pi]) # au/yr

    integrator = Integrator(acceleration)
    ts, qs, ps = integrator.run(q_i=q_i, p_i=p_i,
                                t1=0., t2=10., dt=0.01)

    assert integrator.q_im1.shape == (1,2)
    assert integrator.p_im1.shape == (1,2)

    fig = plot(ts, qs, ps)
    fig.savefig(os.path.join(plot_path,"point_mass_{0}.png".format(name)))

def test_rk5_harmonic_osc():
    T = 10.

    def F(t,x):
        q,p = x.T
        return np.array([p,-(2*np.pi/T)**2*q]).T.copy()

    integrator = RK5Integrator(F)
    #ts, qs, ps = integrator.run(q_i=[[1.],[0.],[0.5]], p_i=[[0.],[1.],[0.5]],
    ts, qs, ps = integrator.run(q_i=np.array([1.]), p_i=np.array([0.]),
                                nsteps=100, dt0=0.1)

    fig = plot(ts, qs, ps)
    fig.savefig(os.path.join(plot_path,"rk5_harm_osc.png"))

def test_rk5_chaotic_pend():
    A = 0.07
    omega_d = 0.75
    def F(t,x):
        q,p = x.T
        return np.array([p,-np.sin(q) + A*np.cos(omega_d*t)]).T.copy()

    integrator = RK5Integrator(F)
    ts, qs, ps = integrator.run(q_i=np.array([3.]), p_i=np.array([0.]),
                                nsteps=10000, dt0=10.)

    #fig = plot(ts, qs, ps)
    fig,ax = plt.subplots(1,1)
    #ax.plot(ts,qs[:,0,0],marker=None)
    ax.plot(qs[:,0,0],ps[:,0,0],marker=None)
    fig.savefig(os.path.join(plot_path,"rk5_pend.png"))