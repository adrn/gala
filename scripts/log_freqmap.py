# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
import gary.dynamics as gd
import gary.potential as gp
import gary.integrate as gi
from gary.units import galactic

def setup_grid(n, potential):
    # grid of points on Phi = 0.5

    # phis = np.linspace(0,2*np.pi,n)
    # thetas = np.arccos(2*np.linspace(0.,1.,n) - 1)
    # HACK
    phis = np.linspace(0.01,1.99*np.pi,n)
    thetas = np.arccos(2*np.linspace(0.01,0.99,n) - 1)
    p,t = np.meshgrid(phis, thetas)
    phis = p.ravel()
    thetas = t.ravel()

    sinp,cosp = np.sin(phis),np.cos(phis)
    sint,cost = np.sin(thetas),np.cos(thetas)

    rh2 = potential.parameters['r_h']**2
    q2 = potential.parameters['q2']
    q3 = potential.parameters['q3']
    r2 = (np.e - rh2) / (sint**2*cosp**2 + sint**2*sinp**2/q2**2 + cost**2/q3**2)
    r = np.sqrt(r2)

    x = r*cosp*sint
    y = r*sinp*sint
    z = r*cost
    v = np.zeros_like(x)

    grid = np.vstack((x,y,z,v,v,v)).T

    return grid

def main():

    # Reproducing Fig. 3.45 from Binney & Tremaine
    potential = gp.LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
                                        q1=1., q2=0.9, q3=0.7, units=galactic)

    dt = 0.05
    nsteps = 2**13

    w0 = setup_grid(25, potential)
    norbits = len(w0)

    # t,ws = potential.integrate_orbit(w0[10], dt=dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    # E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
    # plt.semilogy(np.abs(E[1:]-E[0]))
    # plt.show()
    # gd.plot_orbits(ws, linestyle='none', alpha=0.5)
    # plt.show()

    t,ws = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)

    naff = gd.NAFF(t)
    all_freqs = np.zeros((norbits,3))
    for i in range(norbits):
        f,d,ixes = naff.find_fundamental_frequencies(ws[:,i], nintvec=15)
        all_freqs[i,0] = f[0]
        all_freqs[i,1] = f[1]
        all_freqs[i,2] = f[2]

    np.save("all_freqs.npy", all_freqs)

def plot():
    all_freqs = np.load("all_freqs.npy")

    plt.plot(all_freqs[:,0], all_freqs[:,1], linestyle='none')
    plt.show()

if __name__ == '__main__':

    main()

    # if not os.path.exists("all_freqs.npy"):
    #     main()

    # plot()
