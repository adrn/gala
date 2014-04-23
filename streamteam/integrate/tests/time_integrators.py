# coding: utf-8
"""
    Time the integrators!
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import time

# Third-party
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import G

# Project
from ..leapfrog import LeapfrogIntegrator
from ..rk5 import RK5Integrator
from ..dopri853 import DOPRI853Integrator

GM = (G * (1.*u.M_sun)).decompose([u.au,u.M_sun,u.year,u.radian]).value
timespec = dict(t1=0., t2=10., dt=0.01)

def pointmass_symplectic(Integrator):
    def acceleration(q):
        a = -GM/(q[:,0]**2+q[:,1]**2)**1.5
        return np.array([q[:,0]*a, q[:,1]*a]).T

    q_i = np.array([1.0, 0.0]) # au
    p_i = np.array([0.0, 2*np.pi]) # au/yr

    integrator = Integrator(acceleration)
    ts, qs, ps = integrator.run(q_i=q_i, p_i=p_i, **timespec)

def pointmass_generic(Integrator):
    def F(t,x):
        x,y,px,py = x.T
        a = -GM/(x*x+y*y)**1.5
        return np.array([px, py, x*a, y*a]).T

    q_i = np.array([1.0, 0.0]) # au
    p_i = np.array([0.0, 2*np.pi]) # au/yr

    integrator = Integrator(F)
    ts, xs = integrator.run(x_i=np.append(q_i,p_i), **timespec)

def test():
    ntrials = 10

    a = time.time()
    for ii in range(ntrials):
        pointmass_symplectic(LeapfrogIntegrator)
    print("Leapfrog:",(time.time()-a)/ntrials)

    a = time.time()
    for ii in range(ntrials):
        pointmass_generic(RK5Integrator)
    print("RK5:",(time.time()-a)/ntrials)

    a = time.time()
    for ii in range(ntrials):
        pointmass_generic(DOPRI853Integrator)
    print("DOPRI853I:",(time.time()-a)/ntrials)