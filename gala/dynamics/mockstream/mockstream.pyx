# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

""" Generate mock streams. """


# Standard library
import warnings
from os import path

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
from yaml import dump

from libc.math cimport sqrt
from cpython.exc cimport PyErr_CheckSignals

from ...integrate.cyintegrators.dop853 cimport (dop853_step,
                                                dop853_helper_save_all)
from ...potential.potential.cpotential cimport CPotentialWrapper, CPotential
from ...potential.frame.cframe cimport CFrameWrapper, CFrame
from ...potential.potential.builtin.cybuiltin import NullWrapper

from ...potential import Hamiltonian
from ...potential.frame import StaticFrame
from ...io import quantity_to_hdf5
from ...potential.potential.io import to_dict

from ..nbody.nbody cimport MAX_NBODY
from .df cimport BaseStreamDF

__all__ = ['mockstream_dop853']


cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrame *fr,
                              unsigned norbits, unsigned nbody,
                              void *args) nogil
    void Fwrapper_direct_nbody(unsigned ndim, double t, double *w, double *f,
                               CPotential *p, CFrame *fr,
                               unsigned norbits, unsigned nbody, void *args) nogil


cpdef mockstream_dop853(nbody, double[::1] time,
                        double[:, ::1] stream_w0, double[::1] stream_t1,
                        int[::1] nstream,
                        double atol=1E-10, double rtol=1E-10, int nmax=0):
    """
    Parameters
    ----------
    nbody : `~gala.dynamics.nbody.DirectNBody`
    time : numpy.ndarray (ntimes, )
    stream_w0 : numpy.ndarray (nstreamparticles, 6)
    stream_t1 : numpy.ndarray (ntimes, )
    nstream : numpy.ndarray (ntimes, )
        The number of stream particles to be integrated from this timestep.
        There should be no zero values.

    Notes
    -----
    In code, ``nbodies`` are the massive bodies included from the ``nbody``
    instance passed in. ``nstreamparticles`` are the stream test particles.
    ``nstream`` is the array containing the number of stream particles released
    at each timestep.

    """

    cdef:
        int i, j, k, n # indexing
        unsigned ndim = 6 # TODO: hard-coded, but really must be 6D

        # For N-body support:
        void *args
        CPotential *c_particle_potentials[MAX_NBODY]

        # Time-stepping parameters:
        int ntimes = time.shape[0]
        double dt0 = time[1] - time[0] # initial timestep
        double t2 = time[ntimes-1] # final time

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(nbody.H.potential.c_instance)).cpotential
        CFrame cf = (<CFrameWrapper>(nbody.H.frame.c_instance)).cframe

        # for the test particles
        CPotentialWrapper null_wrapper = NullWrapper(1., [],
                                                     np.zeros(3), np.eye(3))
        CPotential null_p = null_wrapper.cpotential

        int nbodies = nbody._w0.shape[0] # includes the progenitor
        double [:, ::1] nbody_w0 = nbody._w0

        int max_nstream = np.max(nstream)
        int total_nstream = np.sum(nstream)
        double[:, ::1] w_tmp = np.empty((nbodies + max_nstream, ndim))
        double[:, ::1] w_final = np.empty((nbodies + total_nstream, ndim))
        double[:, :, ::1] nbody_w = np.empty((ntimes, nbodies, ndim))

    # set the potential objects of the progenitor (index 0) and any other
    # massive bodies included in the stream generation
    for i in range(nbodies):
        c_particle_potentials[i] = &(<CPotentialWrapper>(nbody.particle_potentials[i].c_instance)).cpotential

    # set null potentials for all of the stream particles
    for i in range(nbodies, nbodies + max_nstream):
        c_particle_potentials[i] = &null_p
    args = <void *>(&c_particle_potentials[0])

    # First have to integrate the nbody orbits so we have their positions at
    # each timestep
    nbody_w = dop853_helper_save_all(&cp, &cf,
                                     <FcnEqDiff> Fwrapper_direct_nbody,
                                     nbody_w0, time,
                                     ndim, nbodies, nbodies, args, ntimes,
                                     atol, rtol, nmax)

    n = 0
    for i in range(ntimes):
        # set initial conditions for progenitor and N-bodies
        for j in range(nbodies):
            for k in range(ndim):
                w_tmp[j, k] = nbody_w[i, j, k]

        for j in range(nstream[i]):
            for k in range(ndim):
                w_tmp[nbodies+j, k] = stream_w0[n+j, k]

        dop853_step(&cp, &cf, <FcnEqDiff> Fwrapper_direct_nbody,
                    &w_tmp[0, 0], stream_t1[i], t2, dt0,
                    ndim, nbodies+nstream[i], nbodies, args,
                    atol, rtol, nmax)

        for j in range(nstream[i]):
            for k in range(ndim):
                w_final[nbodies+n+j, k] = w_tmp[nbodies+j, k]

        PyErr_CheckSignals()

        n += nstream[i]

    for j in range(nbodies):
        for k in range(ndim):
            w_final[j, k] = w_tmp[j, k]

    return_nbody_w = np.array(w_final)[:nbodies]
    return_stream_w = np.array(w_final)[nbodies:]

    return return_nbody_w, return_stream_w
