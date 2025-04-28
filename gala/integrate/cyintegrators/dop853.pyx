# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

""" DOP853 integration in Cython. """

import sys
from libc.stdio cimport *
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy

import numpy as np
cimport numpy as np
np.import_array()

from cpython.exc cimport PyErr_CheckSignals
from ...potential.potential.cpotential cimport CPotentialWrapper
from ...potential.frame.cframe cimport CFrameWrapper

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        pass

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrameType *fr, unsigned norbits,
                              unsigned nbody, void *args) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y,
                              unsigned n, int* irtrn)

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fn,
                CPotential *p, CFrameType *fr, unsigned n_orbits, unsigned nbody,
                void *args,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont,
                Dop853DenseState* dense_state,
                double* tout, unsigned ntout, double* yout)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   CPotential *p, CFrameType *fr, unsigned norbits)

    ctypedef struct Dop853DenseState:
        double *rcont1
        double *rcont2
        double *rcont3
        double *rcont4
        double *rcont5
        double *rcont6
        double *rcont7
        double *rcont8
        double xold
        double hout
        unsigned nrds
        unsigned *indir

    Dop853DenseState* dop853_dense_state_alloc(unsigned nrdens, unsigned n) nogil
    void dop853_dense_state_free(Dop853DenseState* state, unsigned n) nogil
    double contd8_threadsafe(Dop853DenseState *state, unsigned ii, double x) nogil

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

# LEGACY FUNCTION: don't use this (used by lyapunov functionality)
cdef void dop853_step(
    CPotential *cp, CFrameType *cf, FcnEqDiff F,
    double *w, double t1, double t2, double dt0,
    int ndim, int norbits, int nbody, void *args,
    double atol, double rtol, int nmax,
    unsigned err_if_fail,
    unsigned log_output
):

    cdef:
        int res
        SolTrait solout = NULL
        FILE* cfile

    if log_output:
        cfile = stdout
    else:
        cfile = NULL

    res = dop853(
        ndim*norbits, F, cp, cf,
        norbits, nbody, args,
        t1, w, t2,
        &rtol, &atol, 0,  # itoler = 0 for scalar tolerances
        NULL,  # solout: Callback function for output
        0,  # iout: Controls solout call
        cfile,  # fileout: file pointer for logging
        0.0,  # uround: Machine precision (0.0 = use default)
        0.0,  # safe: Safety factor
        0.0,  # fac1: Step size control parameter
        0.0,  # fac2: Step size control parameter
        0.0,  # beta: Stabilizatin for step size control
        0.0,  # hmax: maximum allowed step size
        dt0,  # h: Initial step size
        nmax,  # nmax: maximum number of integration steps
        0,  # meth
        1,  # nstiff: frequency of stiffness detect
        0,  # nrdens: number of components for dense output
        NULL,  # icont: indices for components
        0,  # licont: length of the icont array
        NULL,  # dense_state
        NULL,  # array of output times
        0,  # number of output times
        NULL  # output array for dense output
    )

    if res < 0 and err_if_fail == 1:
        raise RuntimeError(f"Integration failed with code {res}")


cdef class DenseOutputState:
    cdef Dop853DenseState* state
    cdef unsigned n
    def __cinit__(self, unsigned nrdens, unsigned n):
        self.state = dop853_dense_state_alloc(nrdens, n)
        self.n = n
        if self.state is NULL:
            raise MemoryError("Could not allocate Dop853DenseState")
    def __dealloc__(self):
        if self.state is not NULL:
            dop853_dense_state_free(self.state, self.n)

cdef dop853_helper(
    CPotential *cp,
    CFrameType *cf,
    FcnEqDiff F,
    double[:, ::1] w0,
    double[::1] t,
    int ndim,
    int norbits,
    int nbody,
    void *args,
    int ntimes,
    double atol,
    double rtol,
    int nmax,
    double dt_max,
    int nstiff,
    unsigned err_if_fail,
    unsigned log_output,
    unsigned save_all=1
):
    cdef:
        double[:, ::1] w = w0.copy()
        int res
        FILE* cfile

        # Used when save_all = 1
        unsigned size = ndim * norbits
        unsigned nrdens
        DenseOutputState dense_state = DenseOutputState(size, size)
        int ntot = ntimes * norbits * ndim
        double[:, ::1] output_w = np.empty((ntimes, size))
        Dop853DenseState* state
        double* output_ptr

    if save_all:
        output_ptr = &output_w[0, 0]
        state = dense_state.state
        nrdens = size
    else:
        output_ptr = NULL
        state = NULL
        nrdens = 0

    if ntimes < 1:
        raise ValueError("ntimes must be greater than 1")

    if log_output:
        cfile = stdout
    else:
        cfile = NULL

    if w.size != size:
        raise ValueError(f"w0 must be of shape ({norbits}, {ndim}), got size {w.size}")

    res = dop853(
        norbits * ndim, F, cp, cf,
        norbits, nbody, args,
        t[0], &w[0, 0], t[ntimes-1],
        &rtol, &atol, 0,  # itoler = 0 for scalar tolerances, 1 for array
        NULL,  # solout: Callback function for output at each accepted step
        0,  # iout: Controls solout call (0: never, 1: at each step, 2: dense output)
        cfile,  # fileout: file pointer for logging
        np.finfo(float).eps,  # uround: Machine precision
        0.0,  # safe: Safety factor for step size control (0.0 = use default)
        0.0,  # fac1: Step size control parameter (0.0 = use default)
        0.0,  # fac2: Step size control parameter (0.0 = use default)
        0.0,  # beta: Stabilization for step size control (0.0 = use default)
        dt_max,  # hmax: maximum allowed step size (0.0 = no limit)
        t[1] - t[0],  # h: Initial step size
        nmax,  # nmax:  maximum number of integration steps (0 = 100_000)
        1,  # meth:  set to 1 and don't think about it
        nstiff,  # nstiff: frequency of stiffness detect (set to -1 to disable)
        nrdens,  # nrdens: number of components where dense output is needed
        NULL,  # icont: indices for which components get dense out (NULL = all)
        0,  # licont: length of the icont array (ignored if icont=NULL)
        state,
        &t[0],  # array of output times
        ntimes,  # number of output times
        output_ptr  # output array for dense output
    )

    if res < 0 and err_if_fail == 1:
        raise RuntimeError(f"Integration failed with code {res}")

    if save_all:
        return np.asarray(output_w).reshape((ntimes, norbits, ndim))
    else:
        return np.asarray(w).reshape((norbits, ndim))


cpdef dop853_integrate_hamiltonian(
    hamiltonian, double[:, ::1] w0, double[::1] t,
    double atol=1E-10, double rtol=1E-10, int nmax=0, double dt_max = 0.,
    int nstiff=0, int save_all=1, int err_if_fail=1, int log_output=0
):
    """
    CAUTION: Interpretation of axes is different here! We need the
    arrays to be C ordered and easy to iterate over, so here the
    axes are (norbits, ndim).
    """

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    cdef:
        int i, j, k
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]
        void *args

        # define full array of times
        int ntimes = len(t)

        # whoa, so many dots
        CPotential* cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrameType cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

    # 0 below is for nbody - we ignore that in this test particle integration
    w = dop853_helper(
        cp, &cf, <FcnEqDiff> Fwrapper,
        w0, t,
        ndim, norbits, 0, args, ntimes,
        atol, rtol, nmax, dt_max,
        nstiff=nstiff,
        save_all=save_all, err_if_fail=err_if_fail, log_output=log_output
    )
    if save_all:
        return np.asarray(t), np.asarray(w)
    else:
        return np.asarray(t[-1:]), np.asarray(w)
