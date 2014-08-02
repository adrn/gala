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

def main(overwrite=False):

    # only plot up to this index in the orbit plots
    nsteps = 500000
    plot_ix = nsteps//35

    # define an axisymmetric potential
    usys = (u.kpc, u.Msun, u.Myr)
    p = sp.LogarithmicPotential(v_c=0.15, r_h=0., q1=1., q2=1.,
                                q3=0.85, phi=0., usys=usys)

    # initial conditions
    w0 = [8.,0.,0.,0.075,0.15,0.05]

    orbit_filename = os.path.join(plot_path, "orbits.npy")
    action_filename = os.path.join(plot_path, "actions.npy")

    if overwrite:
        if os.path.exists(orbit_filename):
            os.remove(orbit_filename)

        if os.path.exists(action_filename):
            os.remove(action_filename)

    if not os.path.exists(orbit_filename):
        # integrate an orbit in a axisymmetric potential
        acc = lambda t,x: p.acceleration(x)
        integrator = si.LeapfrogIntegrator(acc)
        t,w = integrator.run(w0, dt=1., nsteps=nsteps)

        # also integrate the orbit in the best-fitting isochrone potential
        m,b = sd.fit_isochrone(w, usys=usys)
        isochrone = sp.IsochronePotential(m=m, b=b, usys=usys)
        acc = lambda t,x: isochrone.acceleration(x)
        integrator = si.LeapfrogIntegrator(acc)
        iso_t,iso_w = integrator.run(w0, dt=1., nsteps=plot_ix)

        # cache the orbits
        np.save(orbit_filename, (t,w,iso_t,iso_w))
    else:
        t,w,iso_t,iso_w = np.load(orbit_filename)

    # plot a smaller section of the orbit in projections of XYZ
    fig = sd.plot_orbits(iso_w, marker=None, linestyle='-',
                         alpha=0.5, triangle=True, c='r')
    fig = sd.plot_orbits(w[:plot_ix], axes=fig.axes, marker=None, linestyle='-',
                         alpha=0.8, triangle=True, c='k')
    fig.savefig(os.path.join(plot_path, "orbit_xyz.png"))

    # plot a smaller section of the orbit in the meridional plane
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    R = np.sqrt(w[:,0,0]**2 + w[:,0,1]**2)
    iso_R = np.sqrt(iso_w[:,0,0]**2 + iso_w[:,0,1]**2)
    ax.plot(iso_R, iso_w[:,0,2], marker=None, linestyle='-', alpha=0.5, c='r')
    ax.plot(R[:plot_ix], w[:plot_ix,0,2], marker=None, linestyle='-', alpha=0.8, c='k')
    ax.set_xlabel("$R$")
    ax.set_xlabel("$Z$")
    fig.savefig(os.path.join(plot_path, "orbit_Rz.png"))

    if not os.path.exists(action_filename):
        # compute the actions and angles for the orbit
        actions,angles,freqs = sd.cross_validate_actions(t, w[:,0], N_max=6, nbins=100, usys=usys)

        # now compute for the full time series
        r = find_actions(t, w, N_max, usys, return_Sn=return_Sn)
        full_actions,full_angles,full_freqs = r[:3]
        Sn,dSn_dJ,nvecs = r[3:]

        np.save(action_filename, (actions,angles,freqs) + r)
    else:
        r = np.load(action_filename)
        actions,angles,freqs = r[:3]
        full_actions,full_angles,full_freqs = r[3:6]
        Sn,dSn_dJ,nvecs = r[6:]

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

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-o", dest="overwrite", action="store_true", default=False,
                        help="Overwrite generated files.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(overwrite=args.overwrite)