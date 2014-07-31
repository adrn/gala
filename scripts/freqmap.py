# coding: utf-8

""" Make Figure 9 of Sanders and Binney """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
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

    phis = np.linspace(0,2*np.pi,n)
    thetas = np.arccos(2*np.linspace(0.,1.,n) - 1)
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

def main(n, mpi=False):
    usys = (u.kpc, u.Msun, u.Myr)
    potential = LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
                                     q1=1., q2=0.9, q3=0.7, phi=0.)
    acc = lambda t,x: potential.acceleration(x)
    integrator = LeapfrogIntegrator(acc)

    logger.debug("Setting up grid...")
    grid = setup_grid(n, potential)
    logger.debug("...done!")

    # integrate the orbits
    logger.debug("Integrating orbits...")
    t,w = integrator.run(grid, dt=0.05, nsteps=200000)
    logger.debug("...done!")

    stuffs = zip(np.repeat(t[np.newaxis], len(grid), 0), np.rollaxis(w, 1))
    pool = get_pool(mpi=mpi)

    logger.debug("Computing frequencies...")
    all_freqs = pool.map(worker, stuffs)
    all_freqs = np.array(all_freqs)
    logger.debug("...done!")

    pool.close()

    #np.save("/vega/astro/users/amp2217/projects/new_streamteam/freqs.npy", all_freqs)
    np.save("freqs.npy", all_freqs)

    return
    plt.clf()
    plt.plot(all_freqs[:,1]/all_freqs[:,0], all_freqs[:,2]/all_freqs[:,0],
             linestyle='none', marker=',')
    plt.show()

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

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(n=args.n, mpi=args.mpi)
