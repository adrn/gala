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
import sys

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
from ...potential.frame.cframe cimport CFrameWrapper, CFrameType
from ...potential.potential.builtin.cybuiltin import NullWrapper

from ...potential import Hamiltonian
from ...potential.frame import StaticFrame
from ...io import quantity_to_hdf5
from ...potential.potential.io import to_dict

from ..nbody.nbody cimport MAX_NBODY
from .df cimport BaseStreamDF

__all__ = ['mockstream_dop853', 'mockstream_dop853_animate']


cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrameType *fr,
                              unsigned norbits, unsigned nbody,
                              void *args) nogil
    void Fwrapper_direct_nbody(unsigned ndim, double t, double *w, double *f,
                               CPotential *p, CFrameType *fr,
                               unsigned norbits, unsigned nbody, void *args) nogil


cpdef mockstream_dop853(nbody, double[::1] time,
                        double[:, ::1] stream_w0, double[::1] stream_t1,
                        double tfinal, int[::1] nstream,
                        double atol=1E-10, double rtol=1E-10, int nmax=0,
                        int progress=0):
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

    TODO
    ----
    - `dt0` should be customizable in the Python interface.

    """

    cdef:
        int i, j, k, n  # indexing
        unsigned ndim = 6  # TODO: hard-coded, but really must be 6D

        # For N-body support:
        void *args
        CPotential *c_particle_potentials[MAX_NBODY]

        # Time-stepping parameters:
        int ntimes = time.shape[0]
        double dt0 = 1.

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(nbody.H.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(nbody.H.frame.c_instance)).cframe

        # for the test particles
        CPotentialWrapper null_wrapper = NullWrapper(1., [],
                                                     np.zeros(3), np.eye(3))
        CPotential null_p = null_wrapper.cpotential

        int nbodies = nbody._c_w0.shape[0]  # includes the progenitor
        double [:, ::1] nbody_w0 = nbody._c_w0

        int max_nstream = np.max(nstream)
        int total_nstream = np.sum(nstream)
        double[:, ::1] w_tmp = np.empty((nbodies + max_nstream, ndim))
        double[:, ::1] w_final = np.empty((nbodies + total_nstream, ndim))
        double[:, :, ::1] nbody_w = np.empty((ntimes, nbodies, ndim))

        int prog_out = max(len(time) // 100, 1)

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
                                     atol, rtol, nmax, 0)

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
                    &w_tmp[0, 0], stream_t1[i], tfinal, dt0,
                    ndim, nbodies+nstream[i], nbodies, args,
                    atol, rtol, nmax)

        for j in range(nstream[i]):
            for k in range(ndim):
                w_final[nbodies+n+j, k] = w_tmp[nbodies+j, k]

        PyErr_CheckSignals()

        n += nstream[i]

        if progress == 1:
            if i % prog_out == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    f"Integrating orbits: {100 * i / ntimes: 3.0f}%")
                sys.stdout.flush()

    if progress == 1:
        sys.stdout.write('\r')
        sys.stdout.write(f"Integrating orbits: {100: 3.0f}%")
        sys.stdout.flush()


    for j in range(nbodies):
        for k in range(ndim):
            w_final[j, k] = w_tmp[j, k]

    return_nbody_w = np.array(w_final)[:nbodies]
    return_stream_w = np.array(w_final)[nbodies:]

    return return_nbody_w, return_stream_w


cpdef mockstream_dop853_animate(nbody, double[::1] t,
                                double[:, ::1] stream_w0, int[::1] nstream,
                                output_every=1, output_filename='',
                                overwrite=False, check_filesize=True,
                                double atol=1E-10, double rtol=1E-10,
                                int nmax=0, int progress=0, double dt0=1.):
    """
    Parameters
    ----------
    nbody : `~gala.dynamics.nbody.DirectNBody`
    t : numpy.ndarray (ntimes, )
    stream_w0 : numpy.ndarray (nstreamparticles, 6)
    nstream : numpy.ndarray (ntimes, )
        The number of stream particles to be integrated from this timestep.
        There should be no zero values.

    Notes
    -----
    In code, ``nbodies`` are the massive bodies included from the ``nbody``
    instance passed in. ``nstreamparticles`` are the stream test particles.
    ``nstream`` is the array containing the number of stream particles released
    at each timestep.

    TODO
    ----
    - `dt0` should be customizable in the Python interface.

    """

    cdef:
        int i, j, k, n # indexing
        unsigned ndim = 6 # TODO: hard-coded, but really must be 6D

        # Time-stepping parameters:
        int ntimes = t.shape[0]

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(nbody.H.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(nbody.H.frame.c_instance)).cframe

        int nbodies = nbody._c_w0.shape[0] # includes the progenitor
        double [:, ::1] nbody_w0 = nbody._c_w0

        int total_nstream = np.sum(nstream)
        double[:, ::1] w = np.empty((nbodies + total_nstream, ndim))

        # For N-body support:
        void *args
        CPotential *c_particle_potentials[MAX_NBODY]

        # Snapshotting:
        int noutput_times = (ntimes-1) // output_every + 1
        double[::1] output_times

        int prog_out = max(len(t) // 100, 1)

    if (ntimes-1) % output_every != 0:
        noutput_times += 1 # +1 for final conditions

    output_times = np.zeros(noutput_times)

    est_filesize = total_nstream * noutput_times * 8 * u.byte
    if est_filesize >= 8 * u.gigabyte and check_filesize:
        warnings.warn("Estimated mockstream output file is expected to be "
                      ">8 GB in size! If you're sure, turn this warning "
                      "off with `check_filesize=False`")

    # create the output file
    if path.exists(output_filename) and overwrite == 0:
        raise IOError("Mockstream output file {} already exists! Use "
                      "overwrite=True to overwrite the file."
                      .format(output_filename))

    # set the potential objects of the progenitor (index 0) and any other
    # massive bodies included in the stream generation
    for i in range(nbodies):
        c_particle_potentials[i] = &(<CPotentialWrapper>(nbody.particle_potentials[i].c_instance)).cpotential
    args = <void *>(&c_particle_potentials[0])

    # Initialize the output file:
    import h5py
    h5f = h5py.File(str(output_filename), 'w')
    stream_g = h5f.create_group('stream')
    nbody_g = h5f.create_group('nbody')

    d = stream_g.create_dataset('pos', dtype='f8',
                                shape=(3, noutput_times, total_nstream),
                                fillvalue=np.nan, compression='gzip',
                                compression_opts=9)
    d.attrs['unit'] = str(nbody.units['length'])

    d = stream_g.create_dataset('vel', dtype='f8',
                                shape=(3, noutput_times, total_nstream),
                                fillvalue=np.nan, compression='gzip',
                                compression_opts=9)
    d.attrs['unit'] = str(nbody.units['length'] / nbody.units['time'])

    d = nbody_g.create_dataset('pos', dtype='f8',
                               shape=(3, noutput_times, nbodies),
                               fillvalue=np.nan, compression='gzip',
                               compression_opts=9)
    d.attrs['unit'] = str(nbody.units['length'])

    d = nbody_g.create_dataset('vel', dtype='f8',
                               shape=(3, noutput_times, nbodies),
                               fillvalue=np.nan, compression='gzip',
                               compression_opts=9)
    d.attrs['unit'] = str(nbody.units['length'] / nbody.units['time'])

    # set initial conditions for progenitor and N-bodies
    for j in range(nbodies):
        for k in range(ndim):
            w[j, k] = nbody_w0[j, k]

    for j in range(total_nstream):
        for k in range(ndim):
            w[nbodies+j, k] = stream_w0[j, k]

    n = nstream[0]
    stream_g['pos'][:, 0, :n] = np.array(w[nbodies:nbodies+n, :]).T[:3]
    stream_g['vel'][:, 0, :n] = np.array(w[nbodies:nbodies+n, :]).T[3:]
    nbody_g['pos'][:, 0, :n] = np.array(w[:nbodies, :]).T[:3]
    nbody_g['vel'][:, 0, :n] = np.array(w[:nbodies, :]).T[3:]
    output_times[0] = t[0]

    j = 1 # output time index
    for i in range(1, ntimes):
        # print(i, j, n,
        #       len(t), len(nstream), len(output_times))

        dop853_step(&cp, &cf, <FcnEqDiff> Fwrapper_direct_nbody,
                    &w[0, 0], t[i-1], t[i], dt0,
                    ndim, nbodies+n, nbodies, args,
                    atol, rtol, nmax)

        PyErr_CheckSignals()

        n += nstream[i]

        if (i % output_every) == 0 or i == ntimes-1:
            output_times[j] = t[i]
            stream_g['pos'][:, j, :n] = np.array(w[nbodies:nbodies+n, :]).T[:3]
            stream_g['vel'][:, j, :n] = np.array(w[nbodies:nbodies+n, :]).T[3:]
            nbody_g['pos'][:, j, :n] = np.array(w[:nbodies, :]).T[:3]
            nbody_g['vel'][:, j, :n] = np.array(w[:nbodies, :]).T[3:]
            j += 1

        if progress == 1:
            if i % prog_out == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    f"Integrating orbits: {100 * i / ntimes: 3.0f}%")
                sys.stdout.flush()

    if progress == 1:
        sys.stdout.write('\r')
        sys.stdout.write(f"Integrating orbits: {100: 3.0f}%")
        sys.stdout.flush()

    for g in [stream_g, nbody_g]:
        d = g.create_dataset('time', data=np.array(output_times))
        d.attrs['unit'] = str(nbody.units['time'])

    h5f.close()

    return_nbody_w = np.array(w)[:nbodies]
    return_stream_w = np.array(w)[nbodies:]

    return return_nbody_w, return_stream_w
