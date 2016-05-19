# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" DOP853 integration in Cython. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from libc.stdio cimport printf
from cpython.exc cimport PyErr_CheckSignals

from ...potential.cpotential cimport CPotentialWrapper

cdef extern from "src/cpotential.h":
    ctypedef struct CPotential:
        pass
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, unsigned norbits) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fn,
                CPotential *p, unsigned n_orbits,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   CPotential *p, unsigned norbits)
    double six_norm (double *x)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cdef void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn):
    # TODO: see here for example in FORTRAN:
    #   http://www.unige.ch/~hairer/prog/nonstiff/dr_dop853.f
    pass

cpdef dop853_integrate_potential(CPotentialWrapper cp, double[:,::1] w0,
                                 double[::1] t,
                                 double atol=1E-10, double rtol=1E-10, int nmax=0):
    """
    CAUTION: Interpretation of axes is different here! We need the
    arrays to be C ordered and easy to iterate over, so here the
    axes are (norbits, ndim).

    TODO: add option for a callback function to be called at each step
    """
    cdef:
        int i, j, k
        int res, iout
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]

        # define full array of times
        int ntimes = len(t)
        double dt0 = t[1]-t[0]
        double[::1] w = np.empty(ndim*norbits)

        # Note: icont not needed because nrdens == ndim
        double[:,:,::1] all_w = np.empty((ntimes,norbits,ndim))

    # store initial conditions
    for i in range(norbits):
        for k in range(ndim):
            w[i*ndim + k] = w0[i,k]
            all_w[0,i,k] = w0[i,k]

    # TODO: any way to support dense output?
    iout = 0  # no solout calls

    for j in range(1,ntimes,1):
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     &(cp.cpotential), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, solout, iout,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stiff (interrupted).")

        for k in range(ndim):
            for i in range(norbits):
                all_w[j,i,k] = w[i*ndim + k]

        PyErr_CheckSignals()

    return np.asarray(t), np.asarray(all_w)
