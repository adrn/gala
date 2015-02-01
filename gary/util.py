# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import collections
import sys
import logging
import multiprocessing

# Third-party
import numpy as np

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
        logger.debug("Running with multiprocessing on {} cores..."
                     .format(threads))
        pool = multiprocessing.Pool(threads)

    else:
        logger.debug("Running serial...")
        pool = SerialPool()

    return pool

def gram_schmidt(y):
    """ Modified Gram-Schmidt orthonormalization of the matrix y(n,n) """

    n = y.shape[0]
    if y.shape[1] != n:
        raise ValueError("Invalid shape: {}".format(y.shape))
    mo = np.zeros(n)

    # Main loop
    for i in range(n):
        # Remove component in direction i
        for j in range(i):
            esc = np.sum(y[j]*y[i])
            y[i] -= y[j]*esc

        # Normalization
        mo[i] = np.linalg.norm(y[i])
        y[i] /= mo[i]

    return mo

class use_backend(object):

    def __init__(self, backend):
        import matplotlib.pyplot as plt
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.core.pylabtools import backend2gui

        self.shell = InteractiveShell.instance()
        self.old_backend = backend2gui[str(plt.get_backend())]
        self.new_backend = backend

    def __enter__(self):
        gui, backend = self.shell.enable_matplotlib(self.new_backend)

    def __exit__(self, type, value, tb):
        gui, backend = self.shell.enable_matplotlib(self.old_backend)

def inherit_docs(cls):
    for name, func in vars(cls).items():
        if not func.__doc__:
            for parent in cls.__bases__:
                try:
                    parfunc = getattr(parent, name)
                except AttributeError: # parent doesn't have function
                    break
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls

class ImmutableDict(collections.Mapping):
    def __init__(self, somedict):
        self._dict = dict(somedict)   # make a copy
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def __eq__(self, other):
        return self._dict == other._dict
