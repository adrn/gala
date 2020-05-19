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

# Gala
from gala.units import galactic
from gala.potential.common import PotentialParameter
from gala.potential import PotentialBase
from gala.potential.potential.cpotential cimport (CPotentialWrapper,
                                                  MAX_N_COMPONENTS, CPotential)
from gala.potential.potential.cpotential import CPotentialBase

cdef extern from "extra_compile_macros.h":
    int USE_GSL

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q, int n_dim) nogil
    ctypedef double (*energyfunc)(double t, double *pars, double *q, int n_dim) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, double *grad) nogil

cdef extern from "scf/src/bfe.h":
    double scf_value(double t, double *pars, double *q, int n_dim) nogil
    double scf_density(double t, double *pars, double *q, int n_dim) nogil
    void scf_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double scf_interp_value(double t, double *pars, double *q, int n_dim) nogil
    double scf_interp_density(double t, double *pars, double *q, int n_dim) nogil
    void scf_interp_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

__all__ = ['SCFPotential', 'InterpolatedSCFPotential']


gsl_err_msg = ("Gala was compiled without GSL and so the {classname} class "
               "will not work.  See the gala documentation for more "
               "information about installing and using GSL with gala: "
               "http://gala.adrian.pw/en/latest/install.html")


cdef class SCFWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        if USE_GSL == 1:
            self.cpotential.value[0] = <energyfunc>(scf_value)
            self.cpotential.density[0] = <densityfunc>(scf_density)
            self.cpotential.gradient[0] = <gradientfunc>(scf_gradient)

class SCFPotential(CPotentialBase, GSL_only=True):
    r"""
    SCFPotential(m, r_s, Snlm, Tnlm, units=None, origin=None, R=None)

    A gravitational potential represented as a basis function expansion.  This
    uses the self-consistent field (SCF) method of Hernquist & Ostriker (1992)
    and Lowing et al. (2011), and represents all coefficients as real
    quantities.

    Parameters
    ----------
    m : numeric
        Scale mass.
    r_s : numeric
        Scale length.
    Snlm : array_like
        Array of coefficients for the cos() terms of the expansion.
        This should be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`,
        where `nmax` is the number of radial expansion terms and `lmax`
        is the number of spherical harmonic `l` terms.
    Tnlm : array_like
        Array of coefficients for the sin() terms of the expansion.
        This should be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`,
        where `nmax` is the number of radial expansion terms and `lmax`
        is the number of spherical harmonic `l` terms.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    m = PotentialParameter('m', physical_type='mass')
    r_s = PotentialParameter('r_s', physical_type='length')
    Snlm = PotentialParameter('Snlm', physical_type='dimensionless')
    Tnlm = PotentialParameter('Tnlm', physical_type='dimensionless')

    Wrapper = SCFWrapper

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        PotentialBase.__init__(
            self,
            *args,
            units=units,
            origin=origin,
            R=R,
            **kwargs)

        shp1 = self.parameters['Snlm'].shape
        shp2 = self.parameters['Tnlm'].shape
        if shp1 != shp2:
            raise ValueError(
                "The input coefficient arrays Snlm and Tnlm must have the same "
                f"shape! Received: {shp1} and {shp2}")

        # extra parameters
        nmax = self.parameters['Snlm'].shape[0] - 1
        lmax = self.parameters['Snlm'].shape[1] - 1

        self._setup_wrapper({'nmax': nmax, 'lmax': lmax})


cdef class InterpolatedSCFWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        if USE_GSL == 1:
            self.cpotential.value[0] = <energyfunc>(scf_interp_value)
            self.cpotential.density[0] = <densityfunc>(scf_interp_density)
            self.cpotential.gradient[0] = <gradientfunc>(scf_interp_gradient)


class InterpolatedSCFPotential(CPotentialBase, GSL_only=True):
    r"""
    InterpolatedSCFPotential(m, r_s, Sjnlm, Tjnlm, tj, comj, units=None, origin=None, R=None)

    TODO: update copy-pasta docstring

    A gravitational potential represented as a basis function expansion.  This
    uses the self-consistent field (SCF) method of Hernquist & Ostriker (1992)
    and Lowing et al. (2011), and represents all coefficients as real
    quantities.

    Parameters
    ----------
    m : numeric
        Scale mass.
    r_s : numeric
        Scale length.
    Snlm : array_like
        Array of coefficients for the cos() terms of the expansion.
        This should be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`,
        where `nmax` is the number of radial expansion terms and `lmax`
        is the number of spherical harmonic `l` terms.
    Tnlm : array_like
        Array of coefficients for the sin() terms of the expansion.
        This should be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`,
        where `nmax` is the number of radial expansion terms and `lmax`
        is the number of spherical harmonic `l` terms.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
<<<<<<< HEAD
    m = PotentialParameter('m', physical_type='mass')
    r_s = PotentialParameter('r_s', physical_type='length')
    Sjnlm = PotentialParameter('Sjnlm', physical_type='dimensionless')
    Tjnlm = PotentialParameter('Tjnlm', physical_type='dimensionless')
    tj = PotentialParameter('tj', physical_type='time')
    com_xj = PotentialParameter('com_xj', physical_type='length')
    com_vj = PotentialParameter('com_vj', physical_type='speed')

    Wrapper = InterpolatedSCFWrapper

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        PotentialBase.__init__(
            self,
            *args,
            units=units,
            origin=origin,
            R=R,
            **kwargs)

        shp1 = self.parameters['Sjnlm'].shape
        shp2 = self.parameters['Tjnlm'].shape
        if shp1 != shp2:
            raise ValueError(
                "The input coefficient arrays must have the same "
                f"shape! Received: {shp1} and {shp2}")
=======
    _physical_types = {'m': 'mass',
                       'r_s': 'length',
                       'nmax': 'dimensionless',
                       'lmax': 'dimensionless',
                       'Sjnlm': 'dimensionless',
                       'Tjnlm': 'dimensionless',
                       'tj': 'time',
                       'originj': 'length'}

    def __init__(self, m, r_s, Sjnlm, Tjnlm, tj, originj=None, units=None,
                 R=None, origin=None): # TODO: don't allow passing in these
        from gala._cconfig import GSL_ENABLED
        if not GSL_ENABLED:
            raise ValueError(gsl_err_msg.format('InterpolatedSCFPotential'))

        Sjnlm = np.array(Sjnlm)
        Tjnlm = np.array(Tjnlm)
        tj = np.array(tj)
>>>>>>> TODO

        # extra parameters
        ntimes = self.parameters['Sjnlm'].shape[0]
        nmax = self.parameters['Sjnlm'].shape[1] - 1
        lmax = self.parameters['Sjnlm'].shape[2] - 1

        self._setup_wrapper({'nmax': nmax, 'lmax': lmax, 'ntimes': ntimes})

        # if originj is None:
        #     originj = np.zeros((ntimes, 3))
        # else:
        #     originj = np.array(originj)
