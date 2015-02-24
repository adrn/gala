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

cdef extern from "dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f)
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
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

ctypedef void (*GradFn)(void*, double *params, double *x, double *grad);

# whack, hacky wrapper for gradient function
cdef void F(unsigned ndim, double t, double *w, double *f, GradFn func, double *params):
    cdef int i
    func(NULL, &params[0], &w[0], &f[3])
    for k in range(ndim):
        f[k+ndim] = -f[k+ndim]  # acceleration is minus gradient
        f[k] = w[k+ndim]  # velocities are dw/dx

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

cpdef main(_CPotential cpotential):
    cdef double *w = [0.,1.,0., 0.,0.2,0.]
    cdef double *f = [0.,0.,0.,0.,0.,0.]
    cdef double *params = [4.49975332435e-12, 1E11, 0.5]
    cdef int i

    F(3, 0., &w[0], &f[0], <GradFn>cpotential._gradient2, &params[0])

    print()
    for i in range(6):
        print(f[i])

    # # cdef double[::1] y = np.empty(ndgl)
    # cdef double *y = [2.0, 0.0]
    # cdef int ndgl = 2
    # cdef int res, iout, itoler
    # cdef double x, xend, atoler, rtoler

    # iout = 2
    # x = 0.0
    # xend = 2.0
    # itoler = 0
    # rtoler = 1.0E-6
    # atoler = rtoler

    # print("winding up")

    # res = dop853 (ndgl, fvpol, x, y, xend, &rtoler, &atoler, itoler, solout, iout,
    #     stdout, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0, 0, 1, ndgl, NULL, 0);

    # print("x=xend  y={0:12.10f} {1:12.10f}".format(y[0], y[1]))
    # printf ("rtol=%12.10f   fcn=%li   step=%li   accpt=%li   rejct=%li\r\n",
    #   rtoler, nfcnRead(), nstepRead(), naccptRead(), nrejctRead());

    # return 0;
