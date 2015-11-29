# coding: utf-8
"""
    Test the integrators.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import pytest
import matplotlib.pyplot as pl
import numpy as np

# Project
from .. import LeapfrogIntegrator, RK5Integrator, DOPRI853Integrator

# Integrators to test
integrator_list = [RK5Integrator, DOPRI853Integrator, LeapfrogIntegrator]

# ----------------------------------------------------------------------------

@pytest.mark.parametrize("Integrator", integrator_list)
def test_sho_forward_backward(Integrator):
    def sho(t,w,T):
        q,p = w
        return np.array([p, -(2*np.pi/T)**2*q])

    integrator = Integrator(sho, func_args=(1.,))

    dt = 0.01
    nsteps = 100
    if Integrator == LeapfrogIntegrator:
        dt = 1E-4
        nsteps = int(1E4)

    f_ts, f_ws = integrator.run([0., 1.], dt=dt, nsteps=nsteps)
    b_ts, b_ws = integrator.run([0., 1.], dt=-dt, nsteps=nsteps)

    assert np.allclose(f_ws[:,-1], b_ws[:,-1], atol=1E-6)

@pytest.mark.parametrize("Integrator", integrator_list)
def test_point_mass(Integrator):
    def F(t,w):
        x,y,px,py = w
        a = -1./(x*x+y*y)**1.5
        return np.array([px, py, x*a, y*a])

    q0 = np.array([1., 0.])
    p0 = np.array([0., 1.])
    T = 1.

    integrator = Integrator(F)
    ts, ws = integrator.run(np.append(q0,p0), t1=0., t2=2*np.pi, nsteps=1E4)

    assert np.allclose(ws[:,0], ws[:,-1], atol=1E-6)

@pytest.mark.parametrize("Integrator", integrator_list)
def test_point_mass_multiple(Integrator):
    def F(t,w):
        x,y,px,py = w
        a = -1/(x*x+y*y)**1.5
        return np.array([px, py, x*a, y*a])

    w0 = np.array([[1.0, 0.0, 0.0, 1.],
                   [0.8, 0.0, 0.0, 1.1],
                   [2., 1.0, -1.0, 1.1]]).T

    integrator = Integrator(F)
    ts, ws = integrator.run(w0, dt=1E-3, nsteps=1E4)

@pytest.mark.parametrize("Integrator", integrator_list)
def test_driven_pendulum(Integrator):
    def F(t,w,A,omega_d):
        q,p = w
        return np.array([p,-np.sin(q) + A*np.cos(omega_d*t)])

    integrator = Integrator(F, func_args=(0.07, 0.75))
    ts, ws = integrator.run([3., 0.], dt=1E-2, nsteps=1E4)

@pytest.mark.parametrize("Integrator", integrator_list)
def test_lorenz(Integrator):

    def F(t,w,sigma,rho,beta):
        x,y,z = w
        return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

    sigma, rho, beta = 10., 28., 8/3.
    integrator = Integrator(F, func_args=(sigma, rho, beta))

    if Integrator == LeapfrogIntegrator:
        with pytest.raises(ValueError):
            ts, ws = integrator.run([0.5,0.5,0.5], dt=1E-2, nsteps=1E4)
    else:
        ts, ws = integrator.run([0.5,0.5,0.5], dt=1E-2, nsteps=1E4)

        # pl.plot(ws[0], ws[1])
        # pl.show()

@pytest.mark.parametrize("Integrator", integrator_list)
def test_memmap(tmpdir, Integrator):
    dt = 0.1
    nsteps = 1000
    nw0 = 10000
    mmap = np.memmap("/tmp/test_memmap.npy", mode='w+', shape=(2, nsteps+1, nw0))

    def sho(t,w,T):
        q,p = w
        return np.array([p, -(2*np.pi/T)**2*q])

    w0 = np.random.uniform(-1,1,size=(2,nw0))

    integrator = Integrator(sho, func_args=(1.,))

    ts, ws = integrator.run(w0, dt=dt, nsteps=nsteps, mmap=mmap)
