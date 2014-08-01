# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt

# Package
import streamteam.potential as sp
import streamteam.integrate as si
import streamteam.dynamics as sd

logger.setLevel(logging.DEBUG)

plot_path = os.path.abspath("../_static/dynamics")
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# integrate an orbit in a axisymmetric potential
usys = (u.kpc, u.Msun, u.Myr)
p = sp.MiyamotoNagaiPotential(1E11, 6.5, 0.27, usys=usys)

w0 = [8.,0.,0.,0.075,0.15,0.05]

acc = lambda t,x: p.acceleration(x)
integrator = si.LeapfrogIntegrator(acc)
t,w = integrator.run(w0, dt=1., nsteps=500000)

ss = 35
fig = sd.plot_orbits(w[:len(w)//ss], marker=None, linestyle='-', alpha=0.8, triangle=True)
fig.savefig(os.path.join(plot_path, "orbit_xyz.png"))

fig,ax = plt.subplots(1,1,figsize=(6,6))
R = np.sqrt(w[:,0,0]**2 + w[:,0,1]**2)
ax.plot(R[:len(w)//ss], w[:len(w)//ss,0,2], marker=None, linestyle='-', alpha=0.8)
fig.savefig(os.path.join(plot_path, "orbit_Rz.png"))

actions,angles,freqs = sd.cross_validate_actions(t, w[:,0], N_max=6, nbins=100, usys=usys)

fig,axes = plt.subplots(1,3,figsize=(12,5),sharex=True)
bins = np.linspace(-0.1,0.1,20)
for i in range(3):
    axes[i].set_title("$J_{}$".format(i+1), y=1.02)
    axes[i].hist((actions[:,i] - np.median(actions[:,i])) / np.median(actions[:,i])*100.,
                 bins=bins)
    axes[i].set_xlim(-.11,.11)
    axes[i].set_xticks((-0.1,-0.05,0.,0.05,0.1))
    axes[i].set_xticklabels(("-0.1%","-0.05%","0%","0.05%","0.1%"))

fig.suptitle("Deviation from median", y=0.05, fontsize=18)
fig.tight_layout()
fig.savefig(os.path.join(plot_path, "action_hist.png"))