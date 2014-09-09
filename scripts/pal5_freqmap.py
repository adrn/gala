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
import astropy.coordinates as coord
import astropy.units as u

# Project
import streamteam.coordinates as sc
import streamteam.integrate as si
import streamteam.potential as sp
import streamteam.dynamics as sd
from streamteam.util import get_pool
from streamteam.units import galactic

def setup_grid(n):

    x = [8161.671207,244.760075,16962.073974]*u.pc
    v = [56.429368,101.233841,3.836169]*u.km/u.s

    dv = np.linspace(-50,50,25)*u.km/u.s


    x = x.decompose(galactic).value
    v = v.decompose(galactic).value

    return grid

N_max = 8
def worker(l):
    t = l[:,0]
    w = l[:,1:]
    try:
        # actions,angles,freqs = cross_validate_actions(t, w, N_max=6,
        #                                               usys=None, skip_failures=True)
        actions,angles,freqs = sd.find_actions(t, w, N_max=N_max, usys=None)
        return freqs
    except ValueError as e:
        return None

def parse_batch(batch):
    this_n, n_of = map(int, batch.split("of"))
    return this_n-1, n_of

def main(path, n, mpi=False):
    # has to go here so we don't integrate a huge number of orbits
    pool = get_pool(mpi=mpi)

    # parameters
    N_max = 6
    nsteps = 50000

    usys = (u.kpc, u.Msun, u.Myr)
    potential = sp.LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
                                        q1=1., q2=0.9, q3=0.7, phi=0., usys=usys)
    acc = lambda t,x: potential.acceleration(x)
    integrator = si.LeapfrogIntegrator(acc)

    logger.debug("Setting up grid...")
    grid = setup_grid(n)
    logger.debug("...done!")

    # integrate the orbits
    fn = os.path.join(path, 'orbits.npy')
    if not os.path.exists(fn):
        logger.debug("Integrating orbits...")
        t,w = integrator.run(grid, dt=0.01, nsteps=nsteps)
        logger.debug("...done!")

        NT = 9*N_max**3 # 4 times Sander's value to be safe
        every = nsteps // NT
        t = t[::every]
        w = w[::every]
        l = np.vstack((np.repeat(t.reshape(1,1,t.size), len(grid), axis=1),w.T)).T
        np.save(fn, l)
    else:
        l = np.load(fn)
        t = np.squeeze(l.T[0])
        w = l.T[1:].T
        logger.debug("Read orbits from cache file ({})".format(fn))

    #try:
    #    t = np.repeat(t[np.newaxis], len(grid), 0)
    #    w = np.rollaxis(w,1)
    #    N = np.ones(len(grid),dtype=int)*N_max
    #    stuffs = zip(t, w, N)
    #except:
    #    pool.close()
    #    sys.exit(1)

    logger.debug("Computing frequencies...")
    all_freqs = pool.map(worker, np.rollaxis(l,1))
    all_freqs = np.array(all_freqs)
    logger.debug("...done!")

    pool.close()

    fn = os.path.join(path, "freqs.npy")
    np.save(fn, all_freqs)
    logger.info("Frequencies cached to file:\n\n\t {}".format(fn))

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
    parser.add_argument("--path", dest="path", type=str, required=True,
                        help="Path to cache to.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(path=args.path, n=args.n, mpi=args.mpi)
