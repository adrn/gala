# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

""" Generate mock streams. """


import warnings
from os import path
import sys


import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
from yaml import dump

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cpython.exc cimport PyErr_CheckSignals

from ...integrate.cyintegrators.dop853 cimport dop853_step, dop853_helper, Fwrapper_direct_nbody, FcnEqDiff
from ...integrate.cyintegrators.leapfrog cimport c_init_velocity_nbody, c_leapfrog_step_nbody
from ...potential.potential.cpotential cimport CPotentialWrapper, CPotential, c_gradient, c_nbody_gradient_symplectic
from ...potential.frame.cframe cimport CFrameWrapper, CFrameType
from ...potential.potential.builtin.cybuiltin import NullWrapper

from ...potential import Hamiltonian
from ...potential.frame import StaticFrame
from ...io import quantity_to_hdf5
from ...potential.potential.io import to_dict

from .df cimport BaseStreamDF

__all__ = ['mockstream_dop853', 'mockstream_dop853_animate',
           'mockstream_leapfrog', 'mockstream_leapfrog_animate']


# ==============================================================================
# Helper functions for mockstream implementations

cdef inline CPotential** _setup_particle_potentials(
    nbody, int nbodies, int total_bodies, CPotential* null_p
) except NULL:
    """
    Allocate and initialize particle potentials array.

    Returns pointer to array of CPotential pointers. The first nbodies entries
    point to the massive body potentials, and the remaining entries point to
    the null potential for test particles.
    """
    cdef:
        int i
        CPotential **c_particle_potentials = NULL

    c_particle_potentials = <CPotential**>malloc(total_bodies * sizeof(CPotential*))
    if c_particle_potentials == NULL:
        raise MemoryError("Failed to allocate memory for particle potentials")

    # Set potentials for massive bodies
    for i in range(nbodies):
        c_particle_potentials[i] = (
            <CPotentialWrapper>(nbody.particle_potentials[i].c_instance)
        ).cpotential

    # Set null potentials for test particles
    for i in range(nbodies, total_bodies):
        c_particle_potentials[i] = null_p

    return c_particle_potentials


cdef inline void _report_progress(int i, int ntimes, int prog_out) noexcept nogil:
    """Report integration progress to stdout."""
    if i % prog_out == 0:
        with gil:
            sys.stdout.write('\r')
            sys.stdout.write(
                f"Integrating orbits: {100 * i / ntimes: 3.0f}%")
            sys.stdout.flush()


cdef inline void _finish_progress() noexcept:
    """Finish progress reporting with 100%."""
    sys.stdout.write('\r')
    sys.stdout.write(f"Integrating orbits: {100: 3.0f}%")
    sys.stdout.flush()


cdef inline void _validate_time_arrays(
    int ntimes, int stream_t1_size, int nstream_size
) except *:
    """Validate that time-related arrays have consistent sizes."""
    if stream_t1_size != ntimes:
        raise ValueError("stream_t1 must have the same length as time")
    if nstream_size != ntimes:
        raise ValueError("nstream must have the same length as time")


cdef _init_hdf5_file(
    str output_filename, int overwrite, int check_filesize,
    int noutput_times, int total_nstream, int nbodies, nbody
):
    """
    Initialize HDF5 file for animation output.

    Returns h5py File object with stream and nbody groups and datasets.
    """
    import h5py

    # Check file size estimate
    est_filesize = total_nstream * noutput_times * 8 * u.byte
    if est_filesize >= 8 * u.gigabyte and check_filesize:
        warnings.warn(
            "Estimated mockstream output file is expected to be "
            ">8 GB in size! If you're sure, turn this warning "
            "off with `check_filesize=False`"
        )

    # Check if file exists
    if path.exists(output_filename) and overwrite == 0:
        raise IOError(
            f"Mockstream output file {output_filename} already exists! "
            "Use overwrite=True to overwrite the file."
        )

    # Create file and groups
    h5f = h5py.File(str(output_filename), 'w')
    stream_g = h5f.create_group('stream')
    nbody_g = h5f.create_group('nbody')

    # Create datasets for stream particles
    d = stream_g.create_dataset(
        'pos', dtype='f8',
        shape=(3, noutput_times, total_nstream),
        fillvalue=np.nan, compression='gzip',
        compression_opts=9
    )
    d.attrs['unit'] = str(nbody.units['length'])

    d = stream_g.create_dataset(
        'vel', dtype='f8',
        shape=(3, noutput_times, total_nstream),
        fillvalue=np.nan, compression='gzip',
        compression_opts=9
    )
    d.attrs['unit'] = str(nbody.units['length'] / nbody.units['time'])

    # Create datasets for N-body particles
    d = nbody_g.create_dataset(
        'pos', dtype='f8',
        shape=(3, noutput_times, nbodies),
        fillvalue=np.nan, compression='gzip',
        compression_opts=9
    )
    d.attrs['unit'] = str(nbody.units['length'])

    d = nbody_g.create_dataset(
        'vel', dtype='f8',
        shape=(3, noutput_times, nbodies),
        fillvalue=np.nan, compression='gzip',
        compression_opts=9
    )
    d.attrs['unit'] = str(nbody.units['length'] / nbody.units['time'])

    return h5f, stream_g, nbody_g


cpdef mockstream_dop853(
    nbody, double[::1] time,
    double[:, ::1] stream_w0, double[::1] stream_t1,
    double tfinal, int[::1] nstream,
    double atol=1E-10, double rtol=1E-10, int nmax=0, double dt_max=0.0,
    int nstiff = -1,
    int progress=0,
    int err_if_fail=1, int log_output=0
):
    """
    Parameters
    ----------
    nbody : `~gala.dynamics.nbody.DirectNBody`
    time : numpy.ndarray (ntimes, )
    stream_w0 : numpy.ndarray (nstreamparticles, 6)
    stream_t1 : numpy.ndarray (ntimes, )
    nstream : numpy.ndarray (ntimes, )
        The number of stream particles to be integrated from this timestep.

    Notes
    -----
    In code, ``nbodies`` are the massive bodies included from the ``nbody``
    instance passed in. ``nstreamparticles`` are the stream test particles.
    ``nstream`` is the array containing the number of stream particles released
    at each timestep.

    """

    cdef:
        int i, j, k, n  # indexing
        unsigned ndim = 6  # TODO: hard-coded, but really must be 6D
        void *args
        CPotential **c_particle_potentials = NULL

        # Time-stepping parameters:
        int ntimes = time.shape[0]
        double dt0 = time[1] - time[0]

        CPotential* cp = (<CPotentialWrapper>(nbody.H.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(nbody.H.frame.c_instance)).cframe

        # For test particles
        CPotentialWrapper null_wrapper = NullWrapper(1., [], np.zeros(3), np.eye(3))
        CPotential* null_p = null_wrapper.cpotential

        int nbodies = nbody._c_w0.shape[0]
        double [:, ::1] nbody_w0 = nbody._c_w0

        int total_nstream = np.sum(nstream)
        int total_bodies = nbodies + total_nstream
        double[:, ::1] w_tmp = np.empty((total_bodies, ndim))
        double[:, ::1] w_final = np.empty((total_bodies, ndim))
        double[:, :, ::1] nbody_w = np.empty((ntimes, nbodies, ndim))

        int prog_out = max(len(time) // 100, 1)

    # Validate input arrays
    _validate_time_arrays(ntimes, stream_t1.shape[0], nstream.shape[0])

    # Setup particle potentials
    c_particle_potentials = _setup_particle_potentials(
        nbody, nbodies, total_bodies, null_p
    )
    args = <void *>(c_particle_potentials)

    # TODO: reconfigure this to use dense output?

    try:

        # First have to integrate the nbody orbits so we have their positions at
        # each timestep
        nbody_w = dop853_helper(
            cp, &cf,
            <FcnEqDiff> Fwrapper_direct_nbody,
            nbody_w0, time,
            ndim, nbodies, nbodies, args, ntimes,
            atol, rtol, nmax, dt_max,
            nstiff=nstiff,
            err_if_fail=err_if_fail, log_output=log_output, save_all=1,
        )

        n = 0
        for i in range(ntimes):
            if nstream[i] == 0:
                continue

            # set initial conditions for progenitor and N-bodies
            for j in range(nbodies):
                for k in range(ndim):
                    w_tmp[j, k] = nbody_w[i, j, k]

            for j in range(nstream[i]):
                for k in range(ndim):
                    w_tmp[nbodies+j, k] = stream_w0[n+j, k]

            dop853_step(cp, &cf, <FcnEqDiff> Fwrapper_direct_nbody,
                        &w_tmp[0, 0], stream_t1[i], tfinal, dt0,
                        ndim, nbodies+nstream[i], nbodies, args,
                        atol, rtol, nmax, nstiff=nstiff,
                        err_if_fail=err_if_fail, log_output=log_output)

            PyErr_CheckSignals()

            for j in range(nstream[i]):
                for k in range(ndim):
                    w_final[nbodies+n+j, k] = w_tmp[nbodies+j, k]

            n += nstream[i]

            if progress == 1:
                _report_progress(i, ntimes, prog_out)

        if progress == 1:
            _finish_progress()

        for j in range(nbodies):
            for k in range(ndim):
                w_final[j, k] = w_tmp[j, k]

        return_nbody_w = np.array(w_final)[:nbodies]
        return_stream_w = np.array(w_final)[nbodies:]

        return return_nbody_w, return_stream_w

    finally:
        # Clean up allocated memory
        if c_particle_potentials != NULL:
            free(c_particle_potentials)


cpdef mockstream_dop853_animate(nbody, double[::1] t,
                                double[:, ::1] stream_w0, int[::1] nstream,
                                output_every=1, output_filename='',
                                overwrite=False, check_filesize=True,
                                double atol=1E-10, double rtol=1E-10, int nmax=0,
                                double dt_max=0.0,
                                int nstiff = -1,
                                int progress=0,
                                int err_if_fail=1,
                                int log_output=0):
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

    """

    cdef:
        int i, j, k, n  # indexing
        unsigned ndim = 6  # TODO: hard-coded, but really must be 6D
        void *args
        CPotential **c_particle_potentials = NULL

        # Time-stepping parameters:
        int ntimes = t.shape[0]
        double dt0 = t[1] - t[0]

        CPotential* cp = (<CPotentialWrapper>(nbody.H.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(nbody.H.frame.c_instance)).cframe

        int nbodies = nbody._c_w0.shape[0]
        double [:, ::1] nbody_w0 = nbody._c_w0

        int total_nstream = np.sum(nstream)
        double[:, ::1] w = np.empty((nbodies + total_nstream, ndim))

        # Snapshotting:
        int noutput_times = (ntimes-1) // output_every + 1
        double[::1] output_times

        int prog_out = max(len(t) // 100, 1)

    if (ntimes-1) % output_every != 0:
        noutput_times += 1  # +1 for final conditions

    output_times = np.zeros(noutput_times)

    # Initialize HDF5 output file
    h5f, stream_g, nbody_g = _init_hdf5_file(
        output_filename, overwrite, check_filesize,
        noutput_times, total_nstream, nbodies, nbody
    )

    # Setup particle potentials (only need nbodies, not total_bodies for animate)
    c_particle_potentials = <CPotential**>malloc(nbodies * sizeof(CPotential*))
    if c_particle_potentials == NULL:
        raise MemoryError("Failed to allocate memory for particle potentials")

    try:
        # Set potentials for massive bodies
        for i in range(nbodies):
            c_particle_potentials[i] = (
                <CPotentialWrapper>(nbody.particle_potentials[i].c_instance)
            ).cpotential
        args = <void *>(c_particle_potentials)

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
        nbody_g['pos'][:, 0, :nbodies] = np.array(w[:nbodies, :]).T[:3]
        nbody_g['vel'][:, 0, :nbodies] = np.array(w[:nbodies, :]).T[3:]
        output_times[0] = t[0]

        j = 1 # output time index
        for i in range(1, ntimes):
            dop853_step(cp, &cf, <FcnEqDiff> Fwrapper_direct_nbody,
                        &w[0, 0], t[i-1], t[i], dt0,
                        ndim, nbodies+n, nbodies, args,
                        atol, rtol, nmax, nstiff=nstiff,
                        err_if_fail=err_if_fail, log_output=log_output)

            PyErr_CheckSignals()

            n += nstream[i]

            if (i % output_every) == 0 or i == ntimes-1:
                output_times[j] = t[i]
                stream_g['pos'][:, j, :n] = np.array(w[nbodies:nbodies+n, :]).T[:3]
                stream_g['vel'][:, j, :n] = np.array(w[nbodies:nbodies+n, :]).T[3:]
                nbody_g['pos'][:, j, :nbodies] = np.array(w[:nbodies, :]).T[:3]
                nbody_g['vel'][:, j, :nbodies] = np.array(w[:nbodies, :]).T[3:]
                j += 1

            if progress == 1:
                _report_progress(i, ntimes, prog_out)

        if progress == 1:
            _finish_progress()

        for g in [stream_g, nbody_g]:
            d = g.create_dataset('time', data=np.array(output_times))
            d.attrs['unit'] = str(nbody.units['time'])

        h5f.close()

        return_nbody_w = np.array(w)[:nbodies]
        return_stream_w = np.array(w)[nbodies:]

        return return_nbody_w, return_stream_w

    finally:
        # Clean up allocated memory
        if c_particle_potentials != NULL:
            free(c_particle_potentials)

cpdef mockstream_leapfrog(
    nbody, double[::1] time,
    double[:, ::1] stream_w0, double[::1] stream_t1,
    double tfinal, int[::1] nstream,
    int progress=0,
    int err_if_fail=1
):
    """
    Leapfrog integration version of mockstream generation.

    Parameters
    ----------
    nbody : `~gala.dynamics.nbody.DirectNBody`
    time : numpy.ndarray (ntimes, )
        Times at which stream particles are released.
    stream_w0 : numpy.ndarray (nstreamparticles, 6)
        Initial phase-space coordinates of stream particles.
    stream_t1 : numpy.ndarray (ntimes, )
        Time at which each batch of stream particles is released.
    tfinal : float
        Final time for integration.
    nstream : numpy.ndarray (ntimes, )
        The number of stream particles to be integrated from this timestep.
    progress : int, optional
        Show progress bar (default: 0).
    err_if_fail : int, optional
        Raise error if integration fails (default: 1).

    Notes
    -----
    In code, ``nbodies`` are the massive bodies included from the ``nbody``
    instance passed in. ``nstreamparticles`` are the stream test particles.
    ``nstream`` is the array containing the number of stream particles released
    at each timestep.

    """

    cdef:
        int i, j, k, n, m, t_idx  # indexing
        unsigned ndim = 6  # TODO: hard-coded, but really must be 6D
        int half_ndim = ndim // 2
        CPotential **c_particle_potentials = NULL

        # Time-stepping parameters:
        int ntimes = time.shape[0]
        double dt = time[1] - time[0]

        CPotential* cp = (<CPotentialWrapper>(nbody.H.potential.c_instance)).cpotential

        # For test particles
        CPotentialWrapper null_wrapper = NullWrapper(1., [], np.zeros(3), np.eye(3))
        CPotential* null_p = null_wrapper.cpotential

        int nbodies = nbody._c_w0.shape[0]
        double [:, ::1] nbody_w0 = nbody._c_w0

        int total_nstream = np.sum(nstream)
        int total_bodies = nbodies + total_nstream

        # Working arrays for integration
        double[:, ::1] w_tmp = np.empty((total_bodies, ndim))
        double[:, ::1] w_final = np.empty((total_bodies, ndim))
        double[:, :, ::1] nbody_w = np.empty((ntimes, nbodies, ndim))

        # Leapfrog-specific arrays (half-step velocities)
        double[:, ::1] v_jm1_2 = np.zeros((total_bodies, half_ndim))
        double[::1] grad = np.zeros(half_ndim)

        int n_steps
        int prog_out = max(len(time) // 100, 1)

    # Validate input arrays
    _validate_time_arrays(ntimes, stream_t1.shape[0], nstream.shape[0])

    # Setup particle potentials
    c_particle_potentials = _setup_particle_potentials(
        nbody, nbodies, total_bodies, null_p
    )

    try:

        # STEP 1: Integrate N-body orbits through all timesteps
        # Initialize N-body particle positions
        for i in range(nbodies):
            for k in range(ndim):
                w_tmp[i, k] = nbody_w0[i, k]

        # Initialize half-step velocities for N-body particles
        with nogil:
            for i in range(nbodies):
                for k in range(half_ndim):
                    grad[k] = 0.
                c_init_velocity_nbody(cp, half_ndim, time[0], dt,
                                    c_particle_potentials, &w_tmp[0, 0], nbodies, i,
                                    &w_tmp[i, 0], &w_tmp[i, half_ndim],
                                    &v_jm1_2[i, 0], &grad[0])

        # Save initial N-body state
        for i in range(nbodies):
            for k in range(ndim):
                nbody_w[0, i, k] = w_tmp[i, k]

        # Integrate N-body orbits forward
        with nogil:
            for t_idx in range(1, ntimes):
                for i in range(nbodies):
                    for k in range(half_ndim):
                        grad[k] = 0.
                    c_leapfrog_step_nbody(cp, half_ndim, time[t_idx], dt,
                                        c_particle_potentials, &w_tmp[0, 0], nbodies, i,
                                        &w_tmp[i, 0], &w_tmp[i, half_ndim],
                                        &v_jm1_2[i, 0], &grad[0])

                # Save N-body state at this timestep
                for i in range(nbodies):
                    for k in range(ndim):
                        nbody_w[t_idx, i, k] = w_tmp[i, k]

        # STEP 2: Integrate stream particles from their release times to tfinal
        n = 0  # Counter for total stream particles processed
        for i in range(ntimes):
            if nstream[i] == 0:
                continue

            # Find the time index corresponding to stream_t1[i]
            t_idx = 0
            for j in range(ntimes):
                if time[j] >= stream_t1[i]:
                    t_idx = j
                    break

            # Set initial conditions: N-bodies at release time + new stream particles
            for j in range(nbodies):
                for k in range(ndim):
                    w_tmp[j, k] = nbody_w[t_idx, j, k]

            for j in range(nstream[i]):
                for k in range(ndim):
                    w_tmp[nbodies+j, k] = stream_w0[n+j, k]

            # Initialize half-step velocities for this integration
            with nogil:
                for j in range(nbodies + nstream[i]):
                    for k in range(half_ndim):
                        grad[k] = 0.
                    c_init_velocity_nbody(cp, half_ndim, stream_t1[i], dt,
                                        c_particle_potentials, &w_tmp[0, 0], nbodies, j,
                                        &w_tmp[j, 0], &w_tmp[j, half_ndim],
                                        &v_jm1_2[j, 0], &grad[0])

            # Integrate from stream_t1[i] to tfinal
            n_steps = <int>((tfinal - stream_t1[i]) / dt + 0.5)
            with nogil:
                for j in range(n_steps):
                    for k in range(nbodies + nstream[i]):
                        for m in range(half_ndim):
                            grad[m] = 0.
                        c_leapfrog_step_nbody(cp, half_ndim, stream_t1[i] + (j+1)*dt, dt,
                                            c_particle_potentials, &w_tmp[0, 0], nbodies, k,
                                            &w_tmp[k, 0], &w_tmp[k, half_ndim],
                                            &v_jm1_2[k, 0], &grad[0])

            PyErr_CheckSignals()

            # Save final stream particle state
            for j in range(nstream[i]):
                for k in range(ndim):
                    w_final[nbodies+n+j, k] = w_tmp[nbodies+j, k]

            n += nstream[i]

            if progress == 1:
                _report_progress(i, ntimes, prog_out)

        if progress == 1:
            _finish_progress()

        # Save final N-body positions (from last integration)
        for j in range(nbodies):
            for k in range(ndim):
                w_final[j, k] = w_tmp[j, k]

        return_nbody_w = np.array(w_final)[:nbodies]
        return_stream_w = np.array(w_final)[nbodies:]

        return return_nbody_w, return_stream_w

    finally:
        # Clean up allocated memory
        if c_particle_potentials != NULL:
            free(c_particle_potentials)
