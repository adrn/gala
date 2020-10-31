# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: language_level=3

# Standard library
from libc.math cimport M_PI

# Third party
from astropy.constants import G
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

cdef extern from "extra_compile_macros.h":
    int USE_GSL

cdef extern from "scf/src/bfe_helper.h":
    double rho_nlm(double s, double phi, double X, int n, int l, int m) nogil
    double phi_nlm(double s, double phi, double X, int n, int l, int m) nogil
    double sph_grad_phi_nlm(double s, double phi, double X, int n, int l, int m, double *grad) nogil

cdef extern from "scf/src/bfe.h":
    void scf_density_helper(double *xyz, int K, double M, double r_s,
                            double *Snlm, double *Tnlm,
                            int nmax, int lmax, double *dens) nogil
    void scf_potential_helper(double *xyz, int K, double G, double M, double r_s,
                              double *Snlm, double *Tnlm,
                              int nmax, int lmax, double *potv) nogil
    void scf_gradient_helper(double *xyz, int K, double G, double M, double r_s,
                             double *Snlm, double *Tnlm,
                             int nmax, int lmax, double *grad) nogil

__all__ = ['density', 'potential', 'gradient']

cpdef density(double[:, ::1] xyz,
              double[:, :, ::1] Snlm, double[:, :, ::1] Tnlm,
              double M=1., double r_s=1.):
    """
    density(xyz, Snlm, Tnlm, M=1, r_s=1)

    Compute the density of the basis function expansion
    at a set of positions given the expansion coefficients.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        A 2D array of positions where ``axis=0`` are multiple positions
        and ``axis=1`` are the coordinate dimensions (x, y, z).
    Snlm : `~numpy.ndarray`
        A 3D array of expansion coefficients for the cosine terms
        of the expansion. This notation follows Lowing et al. (2011).
        The array should have shape ``(nmax+1, lmax+1, lmax+1)`` and any
        invalid terms (e.g., when m > l) will be ignored.
    Tnlm : `~numpy.ndarray`
        A 3D array of expansion coefficients for the sine terms
        of the expansion. This notation follows Lowing et al. (2011).
        The array should have shape ``(nmax+1, lmax+1, lmax+1)`` and any
        invalid terms (e.g., when m > l) will be ignored.
    M : numeric (optional)
        Mass scale. Leave unset for dimensionless units.
    r_s : numeric (optional)
        Length scale. Leave unset for dimensionless units.

    Returns
    -------
    dens : `~numpy.ndarray`
        A 1D array of the density at each input position.
        Will have the same length as the input position array, ``xyz``.

    """

    cdef:
        int ncoords = xyz.shape[0]
        double[::1] dens = np.zeros(ncoords)

        int nmax = Snlm.shape[0]-1
        int lmax = Snlm.shape[1]-1

    if USE_GSL == 1:
        scf_density_helper(&xyz[0, 0], ncoords, M, r_s,
                           &Snlm[0, 0, 0], &Tnlm[0, 0, 0],
                           nmax, lmax, &dens[0])

    return np.array(dens)

cpdef potential(double[:, ::1] xyz,
                double[:, :, ::1] Snlm, double[:, :, ::1] Tnlm,
                double G=1., double M=1., double r_s=1.):
    """
    potential(xyz, Snlm, Tnlm, G=1, M=1, r_s=1)

    Compute the gravitational potential of the basis function expansion
    at a set of positions given the expansion coefficients.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        A 2D array of positions where ``axis=0`` are multiple positions
        and ``axis=1`` are the coordinate dimensions (x, y, z).
    Snlm : `~numpy.ndarray`
        A 3D array of expansion coefficients for the cosine terms
        of the expansion. This notation follows Lowing et al. (2011).
        The array should have shape ``(nmax+1, lmax+1, lmax+1)`` and any
        invalid terms (e.g., when m > l) will be ignored.
    Tnlm : `~numpy.ndarray`
        A 3D array of expansion coefficients for the sine terms
        of the expansion. This notation follows Lowing et al. (2011).
        The array should have shape ``(nmax+1, lmax+1, lmax+1)`` and any
        invalid terms (e.g., when m > l) will be ignored.
    G : numeric (optional)
        Gravitational constant. Leave unset for dimensionless units.
    M : numeric (optional)
        Mass scale. Leave unset for dimensionless units.
    r_s : numeric (optional)
        Length scale. Leave unset for dimensionless units.

    Returns
    -------
    pot : `~numpy.ndarray`
        A 1D array of the value of the potential at each input position.
        Will have the same length as the input position array, ``xyz``.

    """
    cdef:
        int ncoords = xyz.shape[0]
        double[::1] potv = np.zeros(ncoords)

        int nmax = Snlm.shape[0]-1
        int lmax = Snlm.shape[1]-1

    if USE_GSL == 1:
        scf_potential_helper(&xyz[0, 0], ncoords, G, M, r_s,
                             &Snlm[0, 0, 0], &Tnlm[0, 0, 0],
                             nmax, lmax, &potv[0])

    return np.array(potv)

cpdef gradient(double[:, ::1] xyz,
               double[:, :, ::1] Snlm, double[:, :, ::1] Tnlm,
               double G=1, double M=1, double r_s=1):
    """
    gradient(xyz, Snlm, Tnlm, G=1, M=1, r_s=1)

    Compute the gradient of the gravitational potential of the
    basis function expansion at a set of positions given the
    expansion coefficients.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        A 2D array of positions where ``axis=0`` are multiple positions
        and ``axis=1`` are the coordinate dimensions (x, y, z).
    Snlm : `~numpy.ndarray`
        A 3D array of expansion coefficients for the cosine terms
        of the expansion. This notation follows Lowing et al. (2011).
        The array should have shape ``(nmax+1, lmax+1, lmax+1)`` and any
        invalid terms (e.g., when m > l) will be ignored.
    Tnlm : `~numpy.ndarray`
        A 3D array of expansion coefficients for the sine terms
        of the expansion. This notation follows Lowing et al. (2011).
        The array should have shape ``(nmax+1, lmax+1, lmax+1)`` and any
        invalid terms (e.g., when m > l) will be ignored.
    G : numeric (optional)
        Gravitational constant. Leave unset for dimensionless units.
    M : numeric (optional)
        Mass scale. Leave unset for dimensionless units.
    r_s : numeric (optional)
        Length scale. Leave unset for dimensionless units.

    Returns
    -------
    grad : `~numpy.ndarray`
        A 2D array of the gradient of the potential at each input position.
        Will have the same shape as the input position array, ``xyz``.

    """
    cdef:
        int ncoords = xyz.shape[0]
        double[:, ::1] grad = np.zeros((ncoords, 3))

        int nmax = Snlm.shape[0]-1
        int lmax = Snlm.shape[1]-1

    if USE_GSL == 1:
        scf_gradient_helper(&xyz[0, 0], ncoords, G, M, r_s,
                            &Snlm[0, 0, 0], &Tnlm[0, 0, 0],
                            nmax, lmax, &grad[0, 0])

    return np.array(grad)
