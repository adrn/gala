# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Standard library
import warnings

# Third-party
from astropy.constants import G
import astropy.units as u

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt
from cpython.exc cimport PyErr_CheckSignals

from ...potential import Hamiltonian, NullPotential
from ...potential.potential.cpotential cimport (CPotentialWrapper,
                                                MAX_N_COMPONENTS, CPotential)
from ...potential.frame.cframe cimport CFrameWrapper
from ...integrate.cyintegrators.dop853 cimport (dop853_helper,
                                                dop853_helper_save_all)

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

cpdef direct_nbody_dop853(double [:, ::1] w0, double[::1] t,
                          hamiltonian, list particle_potentials,
                          save_all=True,
                          double atol=1E-10, double rtol=1E-10, int nmax=0):
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
        CPotential *c_particle_potentials[MAX_NBODY]
        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

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

    # Extract the CPotential objects from the particle potentials.
    for i in range(nparticles):
        c_particle_potentials[i] = &(<CPotentialWrapper>(particle_potentials[i].c_instance)).cpotential

    # We need a void pointer for any other arguments
    args = <void *>(&c_particle_potentials[0])

    if save_all:
        all_w = dop853_helper_save_all(&cp, &cf,
                                       <FcnEqDiff> Fwrapper_direct_nbody,
                                       w0, t,
                                       ndim, nparticles, nbody, args,
                                       ntimes, atol, rtol, nmax, 0)
    else:
        all_w = dop853_helper(&cp, &cf,
                              <FcnEqDiff> Fwrapper_direct_nbody,
                              w0, t,
                              ndim, nparticles, nbody, args, ntimes,
                              atol, rtol, nmax, 0)
        all_w = np.array(all_w).reshape(nparticles, ndim)

    return all_w


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
        CPotential *c_particle_potentials[MAX_NBODY]

        double[:, ::1] acc = np.zeros((nparticles, ps_ndim))

    # Some input validation:
    if len(particle_potentials) != nparticles:
        raise ValueError(
            "The number of particle initial conditions must match the number "
            f"of particle potentials passed in ({nparticles} vs. "
            f"{len(particle_potentials)}).")

    # Extract the CPotential objects from the particle potentials.
    for i in range(nparticles):
        c_particle_potentials[i] = &(<CPotentialWrapper>(particle_potentials[i].c_instance)).cpotential

    c_nbody_acceleration(&c_particle_potentials[0], t, &w0[0, 0],
                         nparticles, nparticles, ndim, &acc[0, 0])

    # NOTES: Just the acceleration, does not handle frames
    return np.asarray(acc)[:, ndim:]
