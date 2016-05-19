# coding: utf-8
# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Generate mock streams. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport M_PI, sqrt
from cpython.exc cimport PyErr_CheckSignals

from ...potential.cpotential cimport CPotentialWrapper
from ._coord cimport (sat_rotation_matrix, to_sat_coords, from_sat_coords,
                      cyl_to_car, car_to_cyl)

__all__ = ['_mock_stream_dop853']

cdef extern from "src/cpotential.h":
    ctypedef struct CPotential:
        pass
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil

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

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cpdef _mock_stream_dop853(CPotentialWrapper cp, double[::1] t, double[:,::1] prog_w,
                          int release_every,
                          _k_mean, _k_disp,
                          double G, _prog_mass,
                          double atol=1E-10, double rtol=1E-10, int nmax=0):
    """
    _mock_stream_dop853(cpotential, t, prog_w, release_every, k_mean, k_disp, G, prog_mass, atol, rtol, nmax)

    Generate a mock stellar stream using the Streakline method.

    Parameters
    ----------
    cpotential : `gala.potential._CPotential`
        An instance of a ``_CPotential`` representing the gravitational potential.
    t : `numpy.ndarray`
        An array of times. Should have shape ``(ntimesteps,)``.
    prog_w : `numpy.ndarray`
        The 6D coordinates for the orbit of the progenitor system at all times.
        Should have shape ``(ntimesteps,6)``.
    release_every : int
        Release particles at the Lagrange points every X timesteps.
    k_mean : `numpy.ndarray`
        Array of mean ``k`` values (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    k_disp : `numpy.ndarray`
        Array of ``k`` value dispersions (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    G : numeric
        The value of the gravitational constant, G, in the unit system used.
    prog_mass : float or `numpy.ndarray`
        The mass of the progenitor or the mass at each time. Should be a scalar or have
        shape ``(ntimesteps,)``.
    atol : numeric (optional)
        Passed to the integrator. Absolute tolerance parameter. Default is 1E-10.
    rtol : numeric (optional)
        Passed to the integrator. Relative tolerance parameter. Default is 1E-10.
    nmax : int (optional)
        Passed to the integrator.
    """
    cdef:
        int i, j, k # indexing
        int res # result from calling dop853
        int ntimes = t.shape[0] # number of times
        int nparticles # total number of test particles released

        unsigned ndim = prog_w.shape[1] # phase-space dimensionality
        unsigned ndim_2 = ndim / 2 # configuration-space dimensionality

        double dt0 = t[1] - t[0] # initial timestep
        double[::1] tmp = np.zeros(3) # temporary array

        double[::1] w_prime = np.zeros(6) # 6-position of stripped star
        double[::1] cyl = np.zeros(6) # 6-position in cylindrical coords
        double[::1] prog_w_prime = np.zeros(6) # 6-position of progenitor rotated
        double[::1] prog_cyl = np.zeros(6) # 6-position of progenitor in cylindrical coords

        # k-factors for parametrized model of Fardal et al. (2015)
        double[::1] ks = np.zeros(6)

        # used for figuring out how many orbits to integrate at any given release time
        unsigned this_ndim, this_norbits

        double Om # angular velocity squared
        double d, sigma_r # distance, dispersion in release positions
        double r_tide, menc, f # tidal radius, mass enclosed, f factor

        double[::1] eps = np.zeros(3) # used for 2nd derivative estimation
        double[:,::1] R = np.zeros((3,3)) # rotation matrix

        double[::1] prog_mass = np.ascontiguousarray(np.atleast_1d(_prog_mass))
        double[:,::1] k_mean = np.ascontiguousarray(np.atleast_2d(_k_mean))
        double[:,::1] k_disp = np.ascontiguousarray(np.atleast_2d(_k_disp))
        double[::1] mu_k
        double[::1] sigma_k

    # figure out how many particles are going to be released into the "stream"
    if ntimes % release_every == 0:
        nparticles = 2 * (ntimes // release_every)
    else:
        nparticles = 2 * (ntimes // release_every + 1)

    # container for only current positions of all particles
    cdef double[::1] w = np.empty(nparticles*ndim)

    # beginning times for each particle
    cdef double[::1] t1 = np.empty(nparticles)
    cdef double t_end = t[ntimes-1]

    # -------

    # copy over initial conditions from progenitor orbit to each streakline star
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        for k in range(ndim):
            w[2*i*ndim + k] = prog_w[j,k]
            w[2*i*ndim + k + ndim] = prog_w[j,k]

        i += 1

    # now go back to each set of initial conditions and modify initial condition
    #   based on mock prescription
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        t1[2*i] = t[j]
        t1[2*i+1] = t[j]

        if prog_mass.shape[0] == 1:
            M = prog_mass[0]
        else:
            M = prog_mass[j]

        if k_mean.shape[0] == 1:
            mu_k = k_mean[0]
            sigma_k = k_disp[0]
        else:
            mu_k = k_mean[j]
            sigma_k = k_disp[j]

        # angular velocity
        d = sqrt(prog_w[j,0]*prog_w[j,0] +
                 prog_w[j,1]*prog_w[j,1] +
                 prog_w[j,2]*prog_w[j,2])
        Om = np.linalg.norm(np.cross(prog_w[j,:3], prog_w[j,3:]) / d**2)

        # gradient of potential in radial direction
        f = Om*Om - c_d2_dr2(&(cp.cpotential), t[j], &prog_w[j,0], &eps[0])
        r_tide = (G*M / f)**(1/3.)

        # the rotation matrix to transform from satellite coords to normal
        sat_rotation_matrix(&prog_w[j,0], &R[0,0])
        to_sat_coords(&prog_w[j,0], &R[0,0], &prog_w_prime[0])
        car_to_cyl(&prog_w_prime[0], &prog_cyl[0])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        # eject stars at tidal radius with same angular velocity as progenitor
        cyl[0] = prog_cyl[0] + ks[0]*r_tide
        cyl[1] = prog_cyl[1] + ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] + ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        cyl[0] = prog_cyl[0] - ks[0]*r_tide
        cyl[1] = prog_cyl[1] - ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] - ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim + ndim])

        i += 1

    for i in range(nparticles):
        res = dop853(ndim, <FcnEqDiff> Fwrapper,
                     &(cp.cpotential), 1,
                     t1[i], &w[i*ndim], t_end, &rtol, &atol, 0, NULL, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

        PyErr_CheckSignals()

    return np.asarray(w).reshape(nparticles, ndim)
