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
from libc.math cimport log

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

cpdef dop853_lyapunov_max(CPotentialWrapper cp, double[::1] w0,
                          double dt, int n_steps, double t0,
                          double d0, int n_steps_per_pullback, int noffset_orbits,
                          double atol=1E-10, double rtol=1E-10, int nmax=0):
    cdef:
        int i, j, k, jiter
        int res
        unsigned ndim = w0.size
        unsigned norbits = noffset_orbits + 1
        unsigned niter = n_steps // n_steps_per_pullback
        double[::1] w = np.empty(norbits*ndim)

        # define full array of times
        double t_end = (<double>n_steps) * dt
        double[::1] t = np.linspace(t0, t_end, n_steps) # TODO: should be n_steps+1

        double d1_mag, norm
        double[:,::1] d1 = np.empty((norbits,ndim))
        double[:,::1] LEs = np.zeros((niter,noffset_orbits))
        double[:,:,::1] all_w = np.zeros((n_steps,norbits,ndim))

        # temp stuff
        double[:,::1] d0_vec = np.random.uniform(size=(noffset_orbits,ndim))

    # store initial conditions
    for i in range(norbits):
        if i == 0:  # store initial conditions for parent orbit
            for k in range(ndim):
                all_w[0,i,k] = w0[k]
                w[i*ndim + k] = all_w[0,i,k]

        else:  # offset orbits
            norm = np.linalg.norm(d0_vec[i-1])
            for k in range(ndim):
                d0_vec[i-1,k] *= d0/norm  # rescale offset vector

                all_w[0,i,k] = w0[k] + d0_vec[i-1,k]
                w[i*ndim + k] = all_w[0,i,k]

    # dummy counter for storing Lyapunov stuff, which only happens every few steps
    jiter = 0
    for j in range(1,n_steps,1):
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     &(cp.cpotential), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, solout, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

        # store position of main orbit
        for i in range(norbits):
            for k in range(ndim):
                all_w[j,i,k] = w[i*ndim + k]

        if (j % n_steps_per_pullback) == 0:
            # get magnitude of deviation vector
            for i in range(1,norbits):
                for k in range(ndim):
                    d1[i,k] = w[i*ndim + k] - w[k]

                d1_mag = six_norm(&d1[i,0])
                LEs[jiter,i-1] = log(d1_mag / d0)

                # renormalize offset orbits
                for k in range(ndim):
                    w[i*ndim + k] = w[k] + d0 * d1[i,k] / d1_mag

            jiter += 1

    LEs = np.array([np.sum(LEs[:j],axis=0)/t[j*n_steps_per_pullback] for j in range(1,niter)])
    return np.asarray(t), np.asarray(all_w), np.asarray(LEs)

cpdef dop853_lyapunov_max_dont_save(CPotentialWrapper cp, double[::1] w0,
                                    double dt, int n_steps, double t0,
                                    double d0, int n_steps_per_pullback, int noffset_orbits,
                                    double atol=1E-10, double rtol=1E-10, int nmax=0):
    cdef:
        int i, j, k, jiter
        int res
        unsigned ndim = w0.size
        unsigned norbits = noffset_orbits + 1
        unsigned niter = n_steps // n_steps_per_pullback
        double[::1] w = np.empty(norbits*ndim)

        # define full array of times
        double t_end = (<double>n_steps) * dt
        double[::1] t = np.linspace(t0, t_end, n_steps) # TODO: should be n_steps+1

        double d1_mag, norm
        double[:,::1] d1 = np.empty((norbits,ndim))
        double[:,::1] LEs = np.zeros((niter,noffset_orbits))

        # temp stuff
        double[:,::1] d0_vec = np.random.uniform(size=(noffset_orbits,ndim))

    # store initial conditions
    for i in range(norbits):
        if i == 0:  # store initial conditions for parent orbit
            for k in range(ndim):
                w[i*ndim + k] = w0[k]

        else:  # offset orbits
            norm = np.linalg.norm(d0_vec[i-1])
            for k in range(ndim):
                d0_vec[i-1,k] *= d0/norm  # rescale offset vector
                w[i*ndim + k] = w0[k] + d0_vec[i-1,k]

    # dummy counter for storing Lyapunov stuff, which only happens every few steps
    jiter = 0
    for j in range(1,n_steps,1):
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     &(cp.cpotential), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, solout, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

        if (j % n_steps_per_pullback) == 0:
            # get magnitude of deviation vector
            for i in range(1,norbits):
                for k in range(ndim):
                    d1[i,k] = w[i*ndim + k] - w[k]

                d1_mag = six_norm(&d1[i,0])
                LEs[jiter,i-1] = log(d1_mag / d0)

                # renormalize offset orbits
                for k in range(ndim):
                    w[i*ndim + k] = w[k] + d0 * d1[i,k] / d1_mag

            jiter += 1

    LEs = np.array([np.sum(LEs[:j],axis=0)/t[j*n_steps_per_pullback] for j in range(1,niter)])
    return np.asarray(LEs)
