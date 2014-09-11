# coding: utf-8

""" Base class for integrators. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging

# Third-party
import numpy as np

# Create logger
logger = logging.getLogger(__name__)

__all__ = ["Integrator"]

class Integrator(object):

    def _prepare_ws(self, w0, mmap):
        """ Decide how to make the return array. If mmap is False, this returns a full array
            of zeros, but with the correct shape as the output. If mmap is True, return a
            pointer to a memory-mapped array. The latter is particularly useful for integrating
            a large number of orbits or integrating a large number of time steps.
        """

        w0 = np.atleast_2d(w0)
        nparticles, ndim = w0.shape

        if ndim % 2 != 0:
            raise ValueError("Dimensionality must be even.")

        # dimensionality of positions,velocities
        self.ndim = ndim
        self.ndim_xv = self.ndim // 2

        x0 = w0[...,:self.ndim_xv]
        v0 = w0[...,self.ndim_xv:]

        return_shape = (nsteps+1,) + w0.shape
        if mmap is None:
            # create the return arrays
            ws = np.zeros(return_shape, dtype=float)

        else:
            if mmap.shape != return_shape:
                raise ValueError("Shape of memory-mapped array doesn't match expected shape of "
                                 "return array ({} vs {})".format(mmap.shape, return_shape))

            if mmap.mode != 'w+':
                raise TypeError("Memory-mapped array must be a writable mode, not '{}'"
                                .format(mmap.mode))

            ws = mmap

        return x0, v0, ws
