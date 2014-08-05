# coding: utf-8

""" Make Figure 9 of Sanders and Binney """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from streamteam.integrate import LeapfrogIntegrator
from streamteam.potential import LogarithmicPotential
from streamteam.dynamics.actionangle import *
from streamteam.util import get_pool

def setup_grid(n, potential):
    # grid of points on Phi = 0.5

    # phis = np.linspace(0,2*np.pi,n)
    # thetas = np.arccos(2*np.linspace(0.,1.,n) - 1)
    # HACK
    phis = np.linspace(0.1,1.95*np.pi,n)
    thetas = np.arccos(2*np.linspace(0.05,0.95,n) - 1)
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

def worker(stuff):
    t,w = stuff
    try:
        actions,angles,freqs = cross_validate_actions(t, w, N_max=6,
                                                      usys=None, skip_failures=True)
        return freqs
    except ValueError as e:
        return None

def parse_batch(batch):
    this_n, n_of = map(int, batch.split("of"))
    return this_n-1, n_of

def main(n, mpi=False, batch=None):
    usys = (u.kpc, u.Msun, u.Myr)
    potential = LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
                                     q1=1., q2=0.9, q3=0.7, phi=0.)
    acc = lambda t,x: potential.acceleration(x)
    integrator = LeapfrogIntegrator(acc)

    # has to go here so we don't integrate a huge number of orbits
    pool = get_pool(mpi=mpi)

    logger.debug("Setting up grid...")
    grid = setup_grid(n, potential)
    logger.debug("...done!")

    if batch is None:
        suffix = ""
    else:
        suffix = "_" + str(batch)
        this_n, n_of = parse_batch(batch)
        grid = grid[this_n::n_of]
        logger.debug("Running batch {} of {}".format(this_n+1, n_of))
        logger.debug("Grid batch length: {}".format(len(grid)))

    # integrate the orbits
    logger.debug("Integrating orbits...")
    t,w = integrator.run(grid, dt=0.05, nsteps=200000)
    logger.debug("...done!")

    t = np.repeat(t[np.newaxis], len(grid), 0)
    w = np.rollaxis(w,1)
    stuffs = zip(t, w)

    logger.debug("Computing frequencies...")
    all_freqs = pool.map(worker, stuffs)
    all_freqs = np.array(all_freqs)
    logger.debug("...done!")

    pool.close()

    #fn = "/vega/astro/users/amp2217/projects/new_streamteam/freqs{}.npy".format(suffix)
    fn = "/tmp/freqs{}.npy".format(suffix)
    np.save(fn, all_freqs)

    # plt.figure(figsize=(6,6))
    # plt.plot(all_freqs[:,1]/all_freqs[:,0], all_freqs[:,2]/all_freqs[:,0],
    #          linestyle='none', marker='.', alpha=0.5)
    # plt.savefig(freqs.png)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--mpi", dest="mpi", action="store_true", default=False,
                        help="Run with MPI.")
    parser.add_argument("-n", dest="n", required=True, type=int,
                        help="Number of elements along one axis of grid.")
    parser.add_argument("-b", "--batch", dest="batch", type=str,
                        help="Batch number, e.g., 1of10.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(n=args.n, mpi=args.mpi, batch=args.batch)
