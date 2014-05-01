# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import multiprocessing

__all__ = ['get_pool']

# Create logger
logger = logging.getLogger(__name__)

class SerialPool(object):

    def close(self):
        return

    def map(self, *args, **kwargs):
        return map(*args, **kwargs)

def get_pool(mpi=False, threads=None):
    """ Get a pool object to pass to emcee for parallel processing.
        If mpi is False and threads is None, pool is None.

        Parameters
        ----------
        mpi : bool
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """

    if mpi:
        from emcee.utils import MPIPool

        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        logger.debug("Running with MPI...")

    elif threads > 1:
        logger.debug("Running with multiprocessing on {} cores..."\
                    .format(threads))
        pool = multiprocessing.Pool(threads)

    else:
        logger.debug("Running serial...")
        pool = SerialPool()

    return pool