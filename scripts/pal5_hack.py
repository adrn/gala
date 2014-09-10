# coding: utf-8

""" Make Figure 9 of Sanders and Binney """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
import streamteam.integrate as si
from streamteam.potential.lm10 import LM10Potential
import streamteam.dynamics as sd
from streamteam.units import galactic

input_path = "/vega/astro/users/amp2217/projects/nonlinear-dynamics/input/pal5"
output_path = "/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/pal5"

def main():
    if not os.path.exists(input_path):
        logger.error("Input path doesn't exist: {}".format(input_path))
        sys.exit(1)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    d = np.loadtxt(os.path.join(input_path, "Pal5_triax_vr_M16_sun.txt"))
    x = (d[:,1:4]*u.pc).decompose(galactic).value
    v = (d[:,4:7]*u.km/u.s).decompose(galactic).value
    w0 = np.hstack((x,v))
    w0 = w0[np.random.randint(len(w0),size=2000)]

    potential = LM10Potential()

    # Integrate orbits and save
    t,w = potential.integrate_orbit(w0, Integrator=si.DOPRI853Integrator,
                                    dt=0.4, nsteps=250000)
    np.save(os.path.join(output_path,"time.npy"), t)
    np.save(os.path.join(output_path,"orbits.npy"), w)

    # Make a few orbit plots
    for ix in np.random.randint(len(w0), size=10):
        fig = sd.plot_orbits(w, ix=ix, alpha=0.01, linestyle='none')
        fig.savefig(os.path.join(output_path, "orbit_{}.png".format(ix)))

    # Make energy conservation check plot
    for i in range(100):
        E = potential.energy(w[...,i,:3], w[...,i,3:])
        plt.clf()
        plt.semilogy(np.abs(E[1:]-E[0])/E[0], marker=None, alpha=0.25)
        plt.ylim(1E-16, 1E-2)
        plt.savefig(os.path.join(output_path, "energy_cons.png"))

    # Compute actions, etc.
    freqs = np.empty((w.shape[1],3))
    angles = np.empty_like(freqs)
    actions = np.empty_like(freqs)
    for i in range(w.shape[1]):
        actions[i],angles[i],freqs[i] = sd.find_actions(t[::10], w[::10,i],
                                                        N_max=6, usys=galactic)

    np.save(os.path.join(output_path,"actions.npy"), actions)
    np.save(os.path.join(output_path,"angles.npy"), angles)
    np.save(os.path.join(output_path,"freqs.npy"), freqs)

    # Make frequency plot
    r1,r2,r3 = np.array([freqs[:,0]-np.mean(freqs[:,0]),
                         freqs[:,1]-np.mean(freqs[:,1]),
                         freqs[:,2]-np.mean(freqs[:,2])])*1000.

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True,sharex=True)
    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(r1[::2], r3[::2], alpha=0.25)
        axes[1].plot(r2[::2], r3[::2], alpha=0.25)

    axes[0].set_xlim(-1.,1.)
    axes[0].set_ylim(-1.,1.)

    axes[0].set_xlabel(r"$\Omega_1-\langle\Omega_1\rangle$ [Gyr$^{-1}$]")
    axes[0].set_ylabel(r"$\Omega_3-\langle\Omega_3\rangle$ [Gyr$^{-1}$]")
    axes[1].set_xlabel(r"$\Omega_2-\langle\Omega_2\rangle$ [Gyr$^{-1}$]")
    fig.savefig(os.path.join(output_path, "frequencies.png"))

    # Make action plot
    r1,r2,r3 = np.array([(actions[:,0]-np.mean(actions[:,0]))/np.mean(actions[:,0]),
                         (actions[:,1]-np.mean(actions[:,1]))/np.mean(actions[:,1]),
                         (actions[:,2]-np.mean(actions[:,2]))/np.mean(actions[:,2])])

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(r1[::2], r3[::2], alpha=0.25)
        axes[1].plot(r2[::2], r3[::2], alpha=0.25)

    axes[0].set_xlabel(r"$(J_1-\langle J_1\rangle)/\langle J_1\rangle$")
    axes[0].set_ylabel(r"$(J_3-\langle J_3\rangle)/\langle J_3\rangle$")
    axes[1].set_xlabel(r"$(J_2-\langle J_2\rangle)/\langle J_2\rangle$")
    fig.savefig(os.path.join(output_path, "actions.png"))

if __name__ == "__main__":
    main()
