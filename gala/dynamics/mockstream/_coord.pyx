# coding: utf-8
# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Coordinate help for generating mock streams. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

from libc.math cimport M_PI

cdef extern from "math.h":
    double fabs(double x) nogil
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double atan2(double y, double x) nogil
    double fmod(double y, double x) nogil

cdef void sat_rotation_matrix(double *w, # in
                              double *R): # out
    cdef:
        double x1_norm, x2_norm, x3_norm = 0.
        unsigned int i
        double *x1 = [0.,0.,0.]
        double *x2 = [0.,0.,0.]
        double *x3 = [0.,0.,0.]

    x1[0] = w[0]
    x1[1] = w[1]
    x1[2] = w[2]

    x3[0] = x1[1]*w[2+3] - x1[2]*w[1+3]
    x3[1] = x1[2]*w[0+3] - x1[0]*w[2+3]
    x3[2] = x1[0]*w[1+3] - x1[1]*w[0+3]

    x2[0] = -x1[1]*x3[2] + x1[2]*x3[1]
    x2[1] = -x1[2]*x3[0] + x1[0]*x3[2]
    x2[2] = -x1[0]*x3[1] + x1[1]*x3[0]

    x1_norm = sqrt(x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2])
    x2_norm = sqrt(x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2])
    x3_norm = sqrt(x3[0]*x3[0] + x3[1]*x3[1] + x3[2]*x3[2])

    for i in range(3):
        x1[i] /= x1_norm
        x2[i] /= x2_norm
        x3[i] /= x3_norm

    R[0] = x1[0]
    R[1] = x1[1]
    R[2] = x1[2]
    R[3] = x2[0]
    R[4] = x2[1]
    R[5] = x2[2]
    R[6] = x3[0]
    R[7] = x3[1]
    R[8] = x3[2]

cdef void to_sat_coords(double *w, double *R, # in
                        double *w_prime): # out
    # Translate to be centered on progenitor
    cdef int i

    # Project into new basis
    w_prime[0] = w[0]*R[0] + w[1]*R[1] + w[2]*R[2]
    w_prime[1] = w[0]*R[3] + w[1]*R[4] + w[2]*R[5]
    w_prime[2] = w[0]*R[6] + w[1]*R[7] + w[2]*R[8]

    w_prime[3] = w[3]*R[0] + w[4]*R[1] + w[5]*R[2]
    w_prime[4] = w[3]*R[3] + w[4]*R[4] + w[5]*R[5]
    w_prime[5] = w[3]*R[6] + w[4]*R[7] + w[5]*R[8]

cdef void from_sat_coords(double *w_prime, double *R, # in
                          double *w): # out
    cdef int i

    # Project back from sat plane
    w[0] = w_prime[0]*R[0] + w_prime[1]*R[3] + w_prime[2]*R[6]
    w[1] = w_prime[0]*R[1] + w_prime[1]*R[4] + w_prime[2]*R[7]
    w[2] = w_prime[0]*R[2] + w_prime[1]*R[5] + w_prime[2]*R[8]

    w[3] = w_prime[3]*R[0] + w_prime[4]*R[3] + w_prime[5]*R[6]
    w[4] = w_prime[3]*R[1] + w_prime[4]*R[4] + w_prime[5]*R[7]
    w[5] = w_prime[3]*R[2] + w_prime[4]*R[5] + w_prime[5]*R[8]

# ---------------------------------------------------------------------

cdef void car_to_cyl(double *w, # in
                     double *cyl): # out
    cdef:
        double R = sqrt(w[0]*w[0] + w[1]*w[1])
        double phi = atan2(w[1], w[0])
        double vR = (w[0]*w[3] + w[1]*w[4]) / R
        double vphi = (w[0]*w[4] - w[3]*w[1]) / R

    cyl[0] = R
    if phi < 0:
        phi = phi + 2*M_PI
    cyl[1] = phi
    cyl[2] = w[2]

    cyl[3] = vR
    cyl[4] = vphi
    cyl[5] = w[5]

cdef void cyl_to_car(double *cyl, # in
                     double *w): # out
    w[0] = cyl[0] * cos(cyl[1])
    w[1] = cyl[0] * sin(cyl[1])
    w[2] = cyl[2]

    w[3] = cyl[3] * cos(cyl[1]) - cyl[4] * sin(cyl[1])
    w[4] = cyl[3] * sin(cyl[1]) + cyl[4] * cos(cyl[1])
    w[5] = cyl[5]

# ---------------------------------------------------------------------
# Tests
#

cpdef _test_sat_rotation_matrix():
    import numpy as np
    np.random.seed(42)
    n = 1024

    cdef:
        double[::1] w = np.zeros(6)
        double[::1] wrot = np.zeros(6)
        double[::1] w2 = np.zeros(6)
        double[:,::1] R = np.zeros((3,3))
        unsigned int i, j

    for i in range(n):
        w = np.random.uniform(size=6)
        sat_rotation_matrix(&w[0], &R[0,0])

        x = np.array(R).dot(np.array(w)[:3])
        assert x[0] > 0
        assert np.allclose(x[1], 0)
        assert np.allclose(x[2], 0)

        v = np.array(R).dot(np.array(w)[3:])
        assert np.allclose(v[2], 0)
        for j in range(3):
            wrot[j] = x[j]
            wrot[j+3] = v[j]

        x2 = np.array(R.T).dot(np.array(wrot)[:3])
        v2 = np.array(R.T).dot(np.array(wrot)[3:])
        for j in range(3):
            w2[j] = x2[j]
            w2[j+3] = v2[j]

        for j in range(6):
            assert np.allclose(w[j], w2[j])

cpdef _test_to_sat_coords_roundtrip():
    import numpy as np
    np.random.seed(42)
    n = 1024

    cdef:
        double[:,::1] w = np.random.uniform(size=(n,6))
        double[:,::1] w_sat = np.random.uniform(size=(n,6))
        double[:,::1] R = np.zeros((3,3))

        double[::1] w_prime = np.zeros(6)
        double[::1] w2 = np.zeros(6)

        unsigned int i, j

    for i in range(n):
        sat_rotation_matrix(&w_sat[i,0], &R[0,0])
        to_sat_coords(&w[i,0], &R[0,0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w2[0])

        for j in range(6):
            assert np.allclose(w[i,j], w2[j])

cpdef _test_car_to_cyl_roundtrip():
    import numpy as np
    np.random.seed(42)
    n = 1024

    cdef:
        double[:,::1] w = np.random.uniform(-10,10,size=(n,6))
        double[::1] cyl = np.zeros(6)
        double[::1] w2 = np.zeros(6)

        unsigned int i, j

    for i in range(n):
        car_to_cyl(&w[i,0], &cyl[0])
        cyl_to_car(&cyl[0], &w2[0])
        for j in range(6):
            assert np.allclose(w[i,j], w2[j])

cpdef _test_cyl_to_car_roundtrip():
    import numpy as np
    # np.random.seed(42)
    n = 1024

    cdef:
        double[:,::1] cyl = np.random.uniform(0,2*np.pi,size=(n,6))
        double[::1] w = np.zeros(6)
        double[::1] cyl2 = np.zeros(6)

        unsigned int i, j

    for i in range(n):
        cyl_to_car(&cyl[i,0], &w[0])
        car_to_cyl(&w[0], &cyl2[0])
        for j in range(6):
            # assert np.allclose(cyl[i,j], cyl2[j])
            if not np.allclose(cyl[i,j], cyl2[j]):
                print(i,j,cyl[i,j], cyl2[j])
        print()


# cdef void car_to_sph(double *xyz, double *sph):
#     # TODO: note this isn't consistent with the velocity transform because of theta
#     # get out spherical components
#     cdef:
#         double d = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2])
#         double phi = atan2(xyz[1], xyz[0])
#         double theta = acos(xyz[2] / d)

#     sph[0] = d
#     sph[1] = phi
#     sph[2] = theta

# cdef void sph_to_car(double *sph, double *xyz):
#     # TODO: note this isn't consistent with the velocity transform because of theta
#     # get out spherical components
#     xyz[0] = sph[0] * cos(sph[1]) * sin(sph[2])
#     xyz[1] = sph[0] * sin(sph[1]) * sin(sph[2])
#     xyz[2] = sph[0] * cos(sph[2])

# cdef void v_car_to_sph(double *xyz, double *vxyz, double *vsph):
#     # get out spherical components
#     cdef:
#         double d = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2])
#         double dxy = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1])

#         double vr = (xyz[0]*vxyz[0]+xyz[1]*vxyz[1]+xyz[2]*vxyz[2]) / d

#         double mu_lon = (xyz[0]*vxyz[1] - vxyz[0]*xyz[1]) / (dxy*dxy)
#         double vlon = mu_lon * dxy # cos(lat)

#         double mu_lat = (xyz[2]*(xyz[0]*vxyz[0] + xyz[1]*vxyz[1]) - dxy*dxy*vxyz[2]) / (d*d*dxy)
#         double vlat = -mu_lat * d

#     vsph[0] = vr
#     vsph[1] = vlon
#     vsph[2] = vlat

# cdef void v_sph_to_car(double *xyz, double *vsph, double *vxyz):
#     # get out spherical components
#     cdef:
#         double d = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2])
#         double dxy = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1])

#     vxyz[0] = vsph[0]*xyz[0]/dxy*dxy/d - xyz[1]/dxy*vsph[1] - xyz[0]/dxy*xyz[2]/d*vsph[2]
#     vxyz[1] = vsph[0]*xyz[1]/dxy*dxy/d + xyz[0]/dxy*vsph[1] - xyz[1]/dxy*xyz[2]/d*vsph[2]
#     vxyz[2] = vsph[0]*xyz[2]/d + dxy/d*vsph[2]

