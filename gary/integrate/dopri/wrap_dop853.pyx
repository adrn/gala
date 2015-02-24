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
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f);
    double contd8 (unsigned ii, double x)

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fcn, double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    long nfcnRead()
    long nstepRead()
    long naccptRead()
    long nrejctRead()
    double hRead()
    double xRead()

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

# custom typedef
ctypedef void (*GradFn)(double *pars, double *q, double *grad);

cdef void F(unsigned ndim, double t, double *w, double *f,
            GradFn func, double *pars):
    cdef int i, half_ndim
    half_ndim = ndim / 2
    print("ndim", ndim)
    print("half_ndim", half_ndim)

    # call gradient function
    func(pars, w, &f[half_ndim])
    print("no f")
    print(f[0],f[1],f[2])
    print(f[3],f[4],f[5])
    print()

    for k in range(half_ndim):
        f[k] = w[k+half_ndim]
        f[k+half_ndim] = -f[k+half_ndim]

cdef void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn):
    pass
    # cdef double xout

    # if nr == 1:
    #     print ("x={0:f}  y={1:12.10f} {2:12.10f}  nstep={3:d}".format(x, y[0], y[1], nr-1))
    #     xout = x + 0.1

    # else:
    #     while (x >= xout):
    #         print ("x={0:f}  y={1:12.10f} {2:12.10f}  nstep={3:d}".format(xout, contd8(0,xout), contd8(1,xout), nr-1))
    #         xout += 0.1

cpdef main(_CPotential cpotential, double[::1] w0):
    cdef int i
    cdef int res, iout, itoler
    cdef double x, xend, atoler, rtoler
    cdef unsigned ndim = 6
    cdef double[::1] w = w0.copy()
    cdef double[::1] f = np.zeros(ndim)

    for i in range(ndim):
        print(w[i], f[i])
    print()

    F(ndim, 0., &w[0], &f[0], <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]))

    for i in range(ndim):
        print(w[i], f[i])


    # iout = 2
    # x = 0.0
    # xend = 1000.0
    # itoler = 0
    # rtoler = 1.0E-6
    # atoler = rtoler

    # print("winding up")

    # res = dop853 (ndim, F, x, &w[0], xend, &rtoler, &atoler, itoler, solout, iout,
    #     stdout, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0, 0, 1, ndim, NULL, 0);

    # print("End w: {}".format(np.array(w)))
    # # print("x=xend  y={0:12.10f} {1:12.10f}".format(y[0], y[1]))
    # # printf ("rtol=%12.10f   fcn=%li   step=%li   accpt=%li   rejct=%li\r\n",
    # #   rtoler, nfcnRead(), nstepRead(), naccptRead(), nrejctRead());

    # # return 0;
