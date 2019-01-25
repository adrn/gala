# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" THIS IS A THIN WRAPPER AROUND THE FUNCTIONS IN coeff_helper.c """

from __future__ import division, print_function

import numpy as np
cimport numpy as np
from libc.math cimport M_PI

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil

cdef extern from "src/coeff_helper.c":
    double c_Snlm_integrand(double phi, double X, double xsi, double density, int n, int l, int m)
    double c_Tnlm_integrand(double phi, double X, double xsi, double density, int n, int l, int m)
    void c_STnlm_discrete(double *s, double *phi, double *X, double *m_k, int K, int n, int l, int m, double *ST)
    void c_STnlm_var_discrete(double *s, double *phi, double *X, double *m_k, int K, int n, int l, int m, double *ST_var)

__all__ = ['Snlm_integrand', 'Tnlm_integrand']

cpdef Snlm_integrand(double phi, double X, double xsi,
                     density_func,
                     int n, int l, int m,
                     double M, double r_s, args):
    cdef:
        double s = (1 + xsi) / (1 - xsi)
        double r = s * r_s
        double x = r * cos(phi) * sqrt(1-X*X)
        double y = r * sin(phi) * sqrt(1-X*X)
        double z = r * X

    return c_Snlm_integrand(phi, X, xsi,
                            density_func(x, y, z, *args) / M * r_s*r_s*r_s,
                            n, l, m)

cpdef Tnlm_integrand(double phi, double X, double xsi,
                     density_func,
                     int n, int l, int m,
                     double M, double r_s, args):
    cdef:
        double s = (1 + xsi) / (1 - xsi)
        double r = s * r_s
        double x = r * cos(phi) * sqrt(1-X*X)
        double y = r * sin(phi) * sqrt(1-X*X)
        double z = r * X

    return c_Tnlm_integrand(phi, X, xsi,
                            density_func(x, y, z, *args) / M * r_s*r_s*r_s,
                            n, l, m)

cpdef STnlm_discrete(double[::1] s, double[::1] phi, double[::1] X,
                     double[::1] m_k,
                     int n, int l, int m):
    cdef:
        double[::1] ST = np.zeros(2)
        int K = s.size

    c_STnlm_discrete(&s[0], &phi[0], &X[0],
                     &m_k[0], K, n, l, m, &ST[0])
    return ST

cpdef STnlm_var_discrete(double[::1] s, double[::1] phi, double[::1] X,
                         double[::1] m_k,
                         int n, int l, int m):
    cdef:
        double[::1] ST_var = np.zeros(2)
        int K = s.size

    c_STnlm_var_discrete(&s[0], &phi[0], &X[0],
                         &m_k[0], K, n, l, m, &ST_var[0])
    return ST_var
