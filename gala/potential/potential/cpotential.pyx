# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Standard library
import copy as pycopy
import sys
import warnings
import uuid

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

from libc.stdio cimport printf

# Project
from .core import PotentialBase, CompositePotential
from ...util import atleast_2d
from ...units import DimensionlessUnitSystem

cdef extern from "math.h":
    double sqrt(double x) nogil
    double fabs(double x) nogil

__all__ = ['CPotentialBase']

cdef extern from "potential/builtin/builtin_potentials.h":
    double nan_density(double t, double *pars, double *q, int n_dim) nogil
    double nan_value(double t, double *pars, double *q, int n_dim) nogil
    void nan_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

cpdef _validate_pos_arr(double[:, ::1] arr):
    if arr.ndim != 2:
        raise ValueError("Phase-space coordinate array must have 2 dimensions")
    return arr.shape[0], arr.shape[1]

cdef class CPotentialWrapper:
    """
    Wrapper class for C implementation of potentials. At the C layer, potentials
    are effectively struct's that maintain pointers to functions specific to a
    given potential. This provides a Cython wrapper around this C implementation.
    """

    cpdef init(self, list parameters, double[::1] q0, double[:, ::1] R,
               int n_dim=3):

        # save the array of parameters so it doesn't get garbage-collected
        self._params = np.array(parameters, dtype=np.float64)

        # an array of number of parameter counts for composite potentials
        self._n_params = np.array([len(self._params)], dtype=np.int32)

        # store pointers to the above arrays
        self.cpotential.n_params = &(self._n_params[0])
        self.cpotential.parameters[0] = &(self._params[0])

        # phase-space half-dimensionality of the potential
        self.cpotential.n_dim = n_dim

        # number of components in the potential. for a simple potential, this is
        #   always one - composite potentials override this.
        self.cpotential.n_components = 1

        # by default, don't skip this potential!
        self.cpotential.null = 0

        # set the function pointers to nan defaults
        self.cpotential.value[0] = <energyfunc>(nan_value)
        self.cpotential.density[0] = <densityfunc>(nan_density)
        self.cpotential.gradient[0] = <gradientfunc>(nan_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(nan_hessian)

        # set the origin of the potentials
        self._q0 = np.array(q0)
        assert len(self._q0) == n_dim
        self.cpotential.q0[0] = &(self._q0[0])

        # set the rotation matrix of the potentials
        self._R = np.ascontiguousarray(np.array(R).ravel())
        self.cpotential.R[0] = &(self._R[0])

    cpdef energy(self, double[:, ::1] q, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double [::1] pot = np.zeros(n)

        if len(t) == 1:
            for i in range(n):
                pot[i] = c_potential(&(self.cpotential), t[0], &q[i, 0])
        else:
            for i in range(n):
                pot[i] = c_potential(&(self.cpotential), t[i], &q[i, 0])

        return np.array(pot)

    cpdef density(self, double[:, ::1] q, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double [::1] dens = np.zeros(n)

        if len(t) == 1:
            for i in range(n):
                dens[i] = c_density(&(self.cpotential), t[0], &q[i, 0])
        else:
            for i in range(n):
                dens[i] = c_density(&(self.cpotential), t[i], &q[i, 0])

        return np.array(dens)

    cpdef gradient(self, double[:, ::1] q, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double[:, ::1] grad = np.zeros((n, ndim))

        if len(t) == 1:
            for i in range(n):
                c_gradient(&(self.cpotential), t[0], &q[i, 0], &grad[i, 0])
        else:
            for i in range(n):
                c_gradient(&(self.cpotential), t[i], &q[i, 0], &grad[i, 0])

        return np.array(grad)

    cpdef hessian(self, double[:, ::1] q, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double[:, :, ::1] hess = np.zeros((n, ndim, ndim))

        if len(t) == 1:
            for i in range(n):
                c_hessian(&(self.cpotential), t[0], &q[i, 0], &hess[i, 0, 0])
        else:
            for i in range(n):
                c_hessian(&(self.cpotential), t[i], &q[i, 0], &hess[i, 0, 0])

        return np.array(hess)

    # ------------------------------------------------------------------------
    # Other functionality
    #
    cpdef d_dr(self, double[:, ::1] q, double G, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double [::1] dr = np.zeros(n, dtype=np.float64)
        cdef double [::1] epsilon = np.zeros(ndim, dtype=np.float64)

        if len(t) == 1:
            for i in range(n):
                dr[i] = c_d_dr(&(self.cpotential), t[0], &q[i, 0], &epsilon[0])
        else:
            for i in range(n):
                dr[i] = c_d_dr(&(self.cpotential), t[i], &q[i, 0], &epsilon[0])

        return np.array(dr)

    cpdef d2_dr2(self, double[:, ::1] q, double G, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double [::1] dr2 = np.zeros(n, dtype=np.float64)
        cdef double [::1] epsilon = np.zeros(ndim, dtype=np.float64)

        if len(t) == 1:
            for i in range(n):
                dr2[i] = c_d2_dr2(&(self.cpotential), t[0], &q[i, 0], &epsilon[0])
        else:
            for i in range(n):
                dr2[i] = c_d2_dr2(&(self.cpotential), t[i], &q[i, 0], &epsilon[0])

        return np.array(dr2)

    cpdef mass_enclosed(self, double[:, ::1] q, double G, double[::1] t):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n, ndim = _validate_pos_arr(q)

        cdef double [::1] mass = np.zeros(n, dtype=np.float64)
        cdef double [::1] epsilon = np.zeros(ndim, dtype=np.float64)

        if len(t) == 1:
            for i in range(n):
                mass[i] = c_mass_enclosed(&(self.cpotential), t[0], &q[i, 0], G, &epsilon[0])
        else:
            for i in range(n):
                mass[i] = c_mass_enclosed(&(self.cpotential), t[i], &q[i, 0], G, &epsilon[0])

        return np.array(mass)

    # For pickling in Python 2
    def __reduce__(self):
        return (self.__class__,
                (self._params[0], list(self._params[1:]),
                 np.array(self._q0),
                 np.array(self._R).reshape(self.cpotential.n_dim,
                                           self.cpotential.n_dim)))

# ----------------------------------------------------------------------------

# TODO: docstrings are now fucked for energy, gradient, etc.

class CPotentialBase(PotentialBase):
    """
    A baseclass for defining gravitational potentials implemented in C.
    """
    Wrapper = None

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        super().__init__(*args,
                         units=units,
                         origin=origin,
                         R=R,
                         **kwargs)
        self._setup_wrapper()

    def _setup_wrapper(self, c_only_parameters=None):
        if self.Wrapper is None:
            raise ValueError("C potential wrapper class not defined for "
                             f"potential class {self.__class__}")

        if c_only_parameters is None:
            c_only_parameters = {}

        arrs = []
        for k, v in c_only_parameters.items():
            arrs.append(np.atleast_1d(v).ravel())

        # to support array parameters, but they get unraveled
        arrs = arrs + [np.atleast_1d(v.value).ravel()
                       for v in self.parameters.values()]

        if len(arrs) > 0:
            self.c_parameters = np.concatenate(arrs)
        else:
            self.c_parameters = np.array([])

        if self.R is None:
            self._R = np.eye(self.ndim)
        else:
            self._R = self.R
        self.c_instance = self.Wrapper(self.G, self.c_parameters,
                                       q0=self.origin, R=self._R)

    def _energy(self, q, t):
        return self.c_instance.energy(q, t=t)

    def _gradient(self, q, t):
        return self.c_instance.gradient(q, t=t)

    def _density(self, q, t):
        return self.c_instance.density(q, t=t)

    def _hessian(self, q, t):
        return self.c_instance.hessian(q, t=t)

    # ----------------------------------------------------------
    # Overwrite the Python potential method to use Cython method
    def mass_enclosed(self, q, t=0.):
        """
        mass_enclosed(q, t)

        Estimate the mass enclosed within the given position by assuming the potential
        is spherical. This is not so good!

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the mass enclosed.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)

        sgn = 1.
        if 'm' in self.parameters and self.parameters['m'] < 0:
            sgn = -1.

        try:
            menc = self.c_instance.mass_enclosed(q, self.G, t=t)
        except AttributeError, TypeError:
            raise ValueError("Potential C instance has no defined "
                             "mass_enclosed function")

        return sgn * menc.reshape(orig_shape[1:]) * self.units['mass']

    def __add__(self, other):
        """
        If all components are Cython, return a CCompositePotential.
        Otherwise, return a standard CompositePotential.
        """
        from .ccompositepotential import CCompositePotential

        if not isinstance(other, PotentialBase):
            raise TypeError('Cannot add a {} to a {}'
                            .format(self.__class__.__name__,
                                    other.__class__.__name__))

        components = dict()

        if isinstance(self, CompositePotential):
            for k, v in self.items():
                components[k] = v

        else:
            k = str(uuid.uuid4())
            components[k] = self

        if isinstance(other, CompositePotential):
            for k, v in self.items():
                if k in components:
                    raise KeyError('Potential component "{}" already exists --'
                                   'duplicate key provided in potential '
                                   'addition')
                components[k] = v

        else:
            k = str(uuid.uuid4())
            components[k] = other

        cython_only = True
        for k, pot in components.items():
            if not isinstance(pot, CPotentialBase):
                cython_only = False
                break

        if cython_only:
            new_pot = CCompositePotential()
        else:
            new_pot = CompositePotential()

        for k, pot in components.items():
            new_pot[k] = pot

        return new_pot

    def replace_units(self, units):
        """Change the unit system of this potential.

        Parameters
        ----------
        units : `~gala.units.UnitSystem`
            Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
        """

        # TODO: this is repeated code - see equivalent in core.py
        tmp = [isinstance(units, DimensionlessUnitSystem),
               isinstance(self.units, DimensionlessUnitSystem)]
        if not all(tmp) and any(tmp):
            raise ValueError("Cannot replace a dimensionless unit system with "
                             "a unit system with physical units, or vice versa")

        return self.__class__(**self.parameters, units=units,
                              R=self.R, origin=self.origin)
