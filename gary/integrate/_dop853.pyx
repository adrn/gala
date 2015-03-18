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

from ..potential.cpotential cimport _CPotential

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil

cdef extern from "dopri/dop853.h":
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
    double six_norm (double *x)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cdef void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn):
    # TODO: see here for example in FORTRAN:
    #   http://www.unige.ch/~hairer/prog/nonstiff/dr_dop853.f
    pass
    # cdef double xout, dx

    # if xold == x:
    #     return

    # print("nr={} xold={} x={} y={} n={}".format(nr, xold, x, y[0], n))

    # # TODO: this is bad - should use a fixed size?
    # dx = (x - xold) / 10.
    # xout = xold + dx
    # while xout <= x:
    #     print("{0:.5f} {1:.5f} {2:.5f} {3:.3f}".format(contd8(0, xout),
    #                                                    contd8(1, xout),
    #                                                    contd8(2, xout),
    #                                                    dx))
    #     xout += dx

cpdef dop853_integrate_potential(_CPotential cpotential, double[:,::1] w0,
                                 double dt0, int nsteps, double t0,
                                 double atol, double rtol, int nmax):
    # TODO: add option for a callback function to be called at each step
    cdef:
        int i, j, k
        int res, iout
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]
        double[::1] t = np.empty(nsteps)
        double[::1] w = np.empty(norbits*ndim)

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

    # define full array of times
    t = np.linspace(t0, t_end, nsteps)
    for j in range(1,nsteps,1):
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, solout, iout,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, nmax, 0, 1, 0, NULL, 0);

        for i in range(norbits):
            for k in range(ndim):
                all_w[j,i,k] = w[i*ndim + k]

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

    return np.asarray(t), np.asarray(all_w)

cpdef dop853_lyapunov(_CPotential cpotential, double[::1] w0,
                      double dt0, int nsteps, double t0,
                      double atol, double rtol,
                      double d0, int nsteps_per_pullback, int noffset_orbits):
    # TODO: add option for a callback function to be called at each step
    cdef:
        int i, j, k, jj
        int res
        unsigned ndim = w0.size
        unsigned norbits = noffset_orbits + 1
        unsigned niter = nsteps // nsteps_per_pullback
        double[::1] t = np.empty(niter)
        double[::1] w = np.empty(norbits*ndim)

        double d1_mag, norm
        double[:,::1] d1 = np.empty((norbits,ndim))
        double[:,::1] LEs = np.zeros((niter,noffset_orbits))
        double[:,::1] main_w = np.zeros((niter+1,ndim))

        # temp stuff
        double[:,::1] d0_vec = np.random.uniform(size=(noffset_orbits,ndim))

    # store initial conditions for parent orbit
    for k in range(ndim):
        w[k] = w0[k]

    # offset vectors to start the offset orbits on - need to be normalized
    for i in range(1,noffset_orbits+1,1):
        norm = np.linalg.norm(d0_vec[i-1])
        for k in range(ndim):
            d0_vec[i-1,k] /= norm
            d0_vec[i-1,k] *= d0
            w[i*ndim + k] = w0[k] + d0_vec[i-1,k]
            main_w[0,k] = w0[k]

    # define full array of times
    time = t0
    for j in range(niter):
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), norbits,
                     time, &w[0], time + dt0*nsteps_per_pullback,
                     &rtol, &atol, 0, solout, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     dt0, 0, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

        # store position of main orbit

        for k in range(ndim):
            main_w[j+1,k] = w[k]

        # get magnitude of deviation vector
        for i in range(1,norbits):
            for k in range(ndim):
                d1[i,k] = w[i*ndim + k] - w[k]

            d1_mag = six_norm(&d1[i,0])
            LEs[j,i-1] = log(d1_mag / d0)

            # renormalize offset orbits
            for k in range(ndim):
                w[i*ndim + k] = w[k] + d0 * d1[i,k] / d1_mag

        # advance time
        time += dt0*nsteps_per_pullback
        t[j] = time

    LEs = np.array([np.sum(LEs[:j],axis=0)/t[j-1] for j in range(1,niter)])
    return np.array(t), np.array(main_w), np.array(LEs)
