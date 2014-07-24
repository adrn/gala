# coding: utf-8

"""
    Implement the Lee & Suto triaxial potential formalism.
    Note: alpha = 3/2 throughout!
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double fabs(double x)
    double exp(double x)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef potential(double[:,::1] _r, double[::1] Phi, int nparticles,
                double vh, double R0, double a, double b, double c):

    cdef double r,x,y,z
    for i in range(nparticles):
        x = _r[i,0]
        y = _r[i,1]
        z = _r[i,2]

        r = sqrt(x*x + y*y + z*z)
        Phi[i] = -vh*vh*(48*R0*a*a*r*r*r*r*log(sqrt(r/R0) + sqrt((R0 + r)/R0)) - 3*R0*(15*R0**2*log(sqrt(r/R0) + sqrt((R0 + r)/R0)) - sqrt(r/(R0 + r))*(15*R0*R0 + 5*R0*r - 2*r*r))*(y*y*(a*a - b*b) + z*z*(a*a - c*c)) - 48*a*a*r*r*r*r*r*(sqrt((R0 + r)/r) - 1) + r*r*(-2*a*a + b*b + c*c)*(-3*R0*(5*R0*R0 - 8*r*r)*log(sqrt(r/R0) + sqrt((R0 + r)/R0)) + 12*r*r*r + sqrt(r/(R0 + r))*(15*R0*R0*R0 + 5*R0*R0*r - 26*R0*r*r - 12*r*r*r)))/(24*a*a*r*r*r*r*r)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef acceleration(double[:,::1] _r, double[:,::1] acc, int nparticles,
                   double vh, double R0, double a, double b, double c):
    cdef:
        double x0 = 1/R0
        double x1 = a*a
        double x2, x3, x4, x5, x6, x7, x8, x9
        double x10 = R0*R0
        double x11, x12, x13, x14
        double x15 = b*b
        double x16 = x1 - x15
        double x17 = c*c
        double x18 = x1 - x17
        double x19
        double x20 = 15*x10
        double x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33
        double x,y,z,r

    for i in range(nparticles):
        x = _r[i,0]
        y = _r[i,1]
        z = _r[i,2]
        r = sqrt(x*x + y*y + z*z)

        x2 = R0 + r
        x3 = 1/x2
        x4 = x0*x2
        x5 = x4**(3/2)
        x6 = sqrt(r*x0)
        x7 = sqrt(x4)
        x8 = x6 + x7
        x9 = vh*vh*x0*x3/(48*r**7*x1*x5*x8)
        x11 = 12*x10*x5*x8
        x12 = r*r*r*r
        x13 = log(x8)
        x14 = 4*r**5*x1*sqrt(x2/r) - 8*x1*x12*x13*x2
        x19 = x16*y**2 + x18*z**2
        x21 = sqrt(r*x3)
        x22 = R0*r
        x23 = r*r
        x24 = -2*x23
        x25 = x21*(x20 + 5*x22 + x24)
        x26 = x2*(x13*x20 - x25)
        x27 = R0*x6*x7 + r
        x28 = x2*x27
        x29 = R0*x7*x8
        x30 = 2*x2*x21
        x31 = 45*x10 + 10*x22
        x32 = 26*x23
        x33 = 3*R0*x19*x2*(-15*R0*x28 + 90*x10*x13*x2*x7*x8 + x25*x29 - x30*x7*x8*(x24 + x31)) + 48*x1*x12*x2**2*x27 - x2*x23*(-2*x1 + x15 + x17)*(R0*x30*x7*x8*(x31 - x32) - 6*x13*x2*x29*(x20 - 8*x23) - x21*x29*(15*R0**3 - R0*x32 - 12*r**3 + 5*r*x10) + x28*(x20 - 24*x23))

        acc[i,0] = x*x9*(x11*(x14 + x19*x26) + x33)
        acc[i,1] = x9*y*(x11*(x14 + x26*(-x16*x23 + x19)) + x33)
        acc[i,2] = x9*z*(x11*(x14 + x26*(-x18*x23 + x19)) + x33)
