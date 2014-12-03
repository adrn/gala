# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

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
from gary.util import get_pool

# timstep and number of steps
dt = 0.02
nsteps = 2**14
nintvec = 15

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

def worker(task):
    i,filename,potential = task
    path = os.path.split(filename)[0]
    freq_fn = os.path.join(path,"{}.npy".format(i))
    if os.path.exists(freq_fn):
        return np.load(freq_fn)

    w0 = np.load(filename)
    t,ws = potential.integrate_orbit(w0[i].copy(), dt=dt, nsteps=nsteps,
                                     Integrator=gi.DOPRI853Integrator)

    naff = gd.NAFF(t)
    try:
        f,d,ixes = naff.find_fundamental_frequencies(ws[:,0], nintvec=nintvec)
    except:
        f = np.array([np.nan,np.nan,np.nan])

    np.save(freq_fn, f)
    return f

    # t,ws = potential.integrate_orbit(w0[10], dt=dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    # E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
    # plt.semilogy(np.abs(E[1:]-E[0]))
    # plt.show()
    # gd.plot_orbits(ws, linestyle='none', alpha=0.5)
    # plt.show()

def main(path="", mpi=False, overwrite=False):
    """ Reproducing Fig. 3.45 from Binney & Tremaine """

    # potential from page 259 in B&T
    potential = gp.LogarithmicPotential(v_c=1., r_h=np.sqrt(0.1),
                                        q1=1., q2=0.9, q3=0.7, units=galactic)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")
    logger.info("Caching to: {}".format(path))
    all_freqs_filename = os.path.join(path, "all_freqs.npy")
    if not os.path.join(path):
        os.mkdir(path)

    # initial conditions
    w0 = setup_grid(100, potential)
    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    # save the initial conditions
    filename = os.path.join(path, 'w0.npy')
    np.save(filename, w0)

    if os.path.exists(all_freqs_filename) and overwrite:
        os.remove(all_freqs_filename)

    if not os.path.exists(all_freqs_filename):
        # for zipping
        filenames = [filename]*norbits
        potentials = [potential]*norbits

        tasks = zip(range(norbits), filenames, potentials)
        all_freqs = pool.map(worker, tasks)
        np.save(all_freqs_filename, np.array(all_freqs))

    pool.close()
    all_freqs = np.load(all_freqs_filename)
    return all_freqs

def plot(freqs, path):
    plt.figure(figsize=(6,6))
    plt.plot(freqs[:,1]/freqs[:,0], freqs[:,2]/freqs[:,0],
             linestyle='none', marker='.', alpha=0.5)
    plt.xlim(0.75, 1.51)
    plt.ylim(1.25, 2.5)
    plt.savefig(os.path.join(path,'freqs.png'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")

    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    parser.add_argument("--path", dest="path", default='', help="Cache path.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    all_freqs = main(path=args.path, mpi=args.mpi, overwrite=args.overwrite)
    plot(freqs=all_freqs, path=args.path)
    sys.exit(0)
