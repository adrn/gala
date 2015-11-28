# coding: utf-8

""" Base class for integrators. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# This project
from ..util import atleast_2d

__all__ = ["Integrator"]

class Integrator(object):

    def _prepare_ws(self, w0, mmap, nsteps):
        """
        Decide how to make the return array. If mmap is False, this returns a
        full array of zeros, but with the correct shape as the output. If mmap
        is True, return a pointer to a memory-mapped array. The latter is
        particularly useful for integrating a large number of orbits or
        integrating a large number of time steps.
        """

        w0 = atleast_2d(w0, insert_axis=1)
        ndim,norbits = w0.shape

        # dimensionality of positions,velocities
        self.ndim = ndim
        self.ndim_xv = self.ndim // 2

        return_shape = (self.ndim,nsteps+1,norbits)
        if mmap is None:
            # create the return arrays
            ws = np.zeros(return_shape, dtype=float)

        else:
            if mmap.shape != return_shape:
                raise ValueError("Shape of memory-mapped array doesn't match expected shape of "
                                 "return array ({} vs {})".format(mmap.shape, return_shape))

            if not mmap.flags.writeable:
                raise TypeError("Memory-mapped array must be a writable mode, not '{}'"
                                .format(mmap.mode))

            ws = mmap

        return w0, ws
