# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3


import warnings


from astropy.constants import G
import astropy.units as u

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cpython.exc cimport PyErr_CheckSignals

from ...potential import Hamiltonian, NullPotential
from ...potential.potential.cpotential cimport CPotentialWrapper, CPotential
from ...potential.frame.cframe cimport CFrameWrapper
from ...integrate.cyintegrators.dop853 cimport dop853_helper

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        pass

cdef extern from "potential/src/cpotential.h":
    void c_nbody_acceleration(CPotential **pots, double t, double *qp,
                              int norbits, int nbody, int ndim, double *acc)

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrameType *fr, unsigned norbits,
                              unsigned nbody, void *args) nogil
    void Fwrapper_direct_nbody(unsigned ndim, double t, double *w, double *f,
                               CPotential *p, CFrameType *fr, unsigned norbits,
                               unsigned nbody, void *args)

cpdef direct_nbody_dop853(
    double [:, ::1] w0, double[::1] t,
    hamiltonian, list particle_potentials,
    save_all=True,
    double atol=1E-10, double rtol=1E-10, int nmax=0, double dt_max=0.0,
    int err_if_fail=1, int log_output=0
):
    """Integrate orbits from initial conditions ``w0`` over the time grid ``t``
    using direct N-body force calculation in the external potential provided via
    the ``hamiltonian`` argument.

    The potential objects for each set of initial conditions must be C-enabled
    (i.e., must be ``CPotentialBase`` subclasses), and the total number of
    potential objects must equal the number of initial conditions.

    By default, this integration procedure stores the full time series of all
    orbits, but this may use a lot of memory. If you just want to store the
    final state of the orbits, pass ``save_all=False``.

    NOTE: This assumes that all massive bodies are organized at the start of w0 and
    particle_potentials, and all test particles are *after* the massive bodies.
    """
    cdef:
        unsigned nparticles = w0.shape[0]
        unsigned nbody = 0
        unsigned ndim = w0.shape[1]
        unsigned ntimes = len(t)

        int i
        void *args
        CPotential **c_particle_potentials = NULL
        CPotential* cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

        double[:, :, ::1] all_w
        double[:, ::1] final_w

    # Some input validation:
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(
            f"Input must be a Hamiltonian object, not {type(hamiltonian)}")

    if not hamiltonian.c_enabled:
        raise TypeError(
            "Input Hamiltonian object does not support C-level access.")

    if len(particle_potentials) != nparticles:
        raise ValueError(
            "The number of particle initial conditions must match the number "
            f"of particle potentials passed in ({nparticles} vs. "
            f"{len(particle_potentials)}).")

    for pot in particle_potentials:
        if not isinstance(pot, NullPotential):
            nbody += 1

    # Dynamically allocate memory for particle potentials
    c_particle_potentials = <CPotential**>malloc(nparticles * sizeof(CPotential*))
    if c_particle_potentials == NULL:
        raise MemoryError("Failed to allocate memory for particle potentials")

    try:
        # Extract the CPotential objects from the particle potentials.
        for i in range(nparticles):
            c_particle_potentials[i] = (<CPotentialWrapper>(particle_potentials[i].c_instance)).cpotential

        # We need a void pointer for any other arguments
        args = <void *>(c_particle_potentials)

        w = dop853_helper(
            cp, &cf,
            <FcnEqDiff> Fwrapper_direct_nbody,
            w0, t,
            ndim, nparticles, nbody, args,
            ntimes,
            atol, rtol, nmax, dt_max,
            nstiff=-1,  # disable stiffness check - TODO: note somewhere
            err_if_fail=err_if_fail, log_output=log_output,
            save_all=save_all
        )
        if save_all:
            return np.array(w)
        else:
            return np.array(w).reshape(nparticles, ndim)

    finally:
        # Clean up allocated memory
        if c_particle_potentials != NULL:
            free(c_particle_potentials)


cpdef nbody_acceleration(double [:, ::1] w0, double t,
                         list particle_potentials):
    """
    Computes the N-body acceleration on a set of bodies at phase-space
    positions w0.
    """
    cdef:
        unsigned nparticles = w0.shape[0]
        unsigned ps_ndim = w0.shape[1]
        unsigned ndim = ps_ndim // 2

        int i
        CPotential **c_particle_potentials = NULL
        double[:, ::1] acc = np.zeros((nparticles, ps_ndim))

    # Some input validation:
    if len(particle_potentials) != nparticles:
        raise ValueError(
            "The number of particle initial conditions must match the number "
            f"of particle potentials passed in ({nparticles} vs. "
            f"{len(particle_potentials)}).")

    # Dynamically allocate memory for particle potentials
    c_particle_potentials = <CPotential**>malloc(nparticles * sizeof(CPotential*))
    if c_particle_potentials == NULL:
        raise MemoryError("Failed to allocate memory for particle potentials")

    try:
        # Extract the CPotential objects from the particle potentials.
        for i in range(nparticles):
            c_particle_potentials[i] = (<CPotentialWrapper>(particle_potentials[i].c_instance)).cpotential

        c_nbody_acceleration(c_particle_potentials, t, &w0[0, 0],
                            nparticles, nparticles, ndim, &acc[0, 0])

        # NOTES: Just the acceleration, does not handle frames
        return np.asarray(acc)[:, ndim:]

    finally:
        # Clean up allocated memory
        if c_particle_potentials != NULL:
            free(c_particle_potentials)
