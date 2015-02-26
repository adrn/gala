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

from ...potential.cpotential cimport _CPotential

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef extern from "dop853.h":
    ctypedef void (*GradFn)(double *pars, double *q, double *grad) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f, GradFn gradfunc, double *gpars, unsigned norbits) nogil
    double contd8 (unsigned ii, double x)

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fcn, GradFn gradfunc, double *gpars, unsigned norbits,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   GradFn func, double *pars, unsigned norbits)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cdef void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn):
    # TODO: see here for example in FORTRAN:
    #   http://www.unige.ch/~hairer/prog/nonstiff/dr_dop853.f
    cdef double xout, dx

    if xold == x:
        return

    print("nr={} xold={} x={} y={} n={}".format(nr, xold, x, y[0], n))

    # TODO: this is bad - should use a fixed size?
    dx = (x - xold) / 10.
    xout = xold + dx
    while xout <= x:
        print("{0:.5f} {1:.5f} {2:.5f} {3:.3f}".format(contd8(0, xout),
                                                       contd8(1, xout),
                                                       contd8(2, xout),
                                                       dx))
        xout += dx

cpdef dop853_integrate_potential(_CPotential cpotential, double[:,::1] w0,
                                 double dt0, int nsteps, double t0,
                                 double atol, double rtol):
    # TODO: add option for a callback function to be called at each step
    cdef:
        int i, j, k
        int res, iout
        unsigned ptr
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]
        double[::1] t = np.empty(nsteps)
        double[::1] w = np.empty(norbits*ndim)
        double[::1] f = np.empty(norbits*ndim)

        # Note: icont not needed because nrdens == ndim
        double t_end = (<double>nsteps) * dt0
        double[:,:,::1] all_w = np.empty((nsteps,norbits,ndim))

    # store initial conditions
    for i in range(norbits):
        for k in range(ndim):
            w[i*ndim + k] = w0[i,k]
            all_w[0,i,k] = w0[i,k]

    # TODO: dense output?
    iout = 0  # no solout calls
    # iout = 2  # dense output

    # F(ndim, 0., &w[0], &f[0],
    #   <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]))

    print("address of cpot", hex(id(cpotential)))
    print("address of w", hex(id(w)))
    print("address of w0", hex(id(w0)))
    print("address of f", hex(id(f)))

    # define full array of times
    t = np.linspace(t0, t_end, nsteps)
    for j in range(1,nsteps,1):
        print(j)
        res = dop853(ndim, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, solout, iout,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0, 0, 1, ndim, NULL, ndim);

        # for i in range(norbits):
        #     for k in range(ndim):
        #         all_w[j,i,k] = w[i*ndim + k]

    if res == -1:
        raise RuntimeError("Input is not consistent.")
    elif res == -2:
        raise RuntimeError("Larger nmax is needed.")
    elif res == -3:
        raise RuntimeError("Step size becomes too small.")
    elif res == -4:
        raise RuntimeError("The problem is probably stff (interrupted).")

    return np.asarray(t), np.asarray(all_w)
