# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t

from ..cpotential cimport CPotentialWrapper
from ..cpotential cimport densityfunc, energyfunc, gradientfunc, hessianfunc
from ...._cconfig cimport USE_GSL

# Time interpolation state structure and GSL types (declared in time_interp.h)
cdef extern from "time_interp.h":
    # Forward declaration of GSL type (defined in time_interp.h)
    ctypedef struct gsl_interp_type:
        pass

    # GSL interpolation type pointers - these will only link if USE_GSL==1
    gsl_interp_type * gsl_interp_linear
    gsl_interp_type * gsl_interp_cspline
    gsl_interp_type * gsl_interp_akima
    gsl_interp_type * gsl_interp_steffen

cdef extern from "time_interp.h":
    ctypedef struct TimeInterpParam:
        int is_constant
        double constant_value
        int n_knots

    ctypedef struct TimeInterpRotation:
        int is_constant
        double constant_matrix[9]

    ctypedef struct TimeInterpState:
        TimeInterpParam *params
        TimeInterpParam origin
        TimeInterpRotation rotation
        int n_params
        int n_dim
        const gsl_interp_type *interp_type
        double t_min
        double t_max
        void *wrapped_potential

    TimeInterpState* time_interp_alloc(int n_params, int n_dim, const gsl_interp_type *interp_type)
    void time_interp_free(TimeInterpState *state)
    int time_interp_init_param(TimeInterpParam *param, double *time_knots, double *values,
                              int n_knots, int n_elements, const gsl_interp_type *interp_type)
    int time_interp_init_constant_param(TimeInterpParam *param, double *constant_values, int n_elements)
    int time_interp_init_rotation(TimeInterpRotation *rot, double *time_knots, double *matrices,
                                 int n_knots, const gsl_interp_type *interp_type)
    int time_interp_init_constant_rotation(TimeInterpRotation *rot, double *matrix)

cdef extern from "time_interp_wrapper.h":
    double time_interp_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void time_interp_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    double time_interp_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void time_interp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

__all__ = ['TimeInterpolatedWrapper']


cdef class TimeInterpolatedWrapper(CPotentialWrapper):
    """
    Cython wrapper for time-interpolated potentials.
    """
    cdef TimeInterpState *interp_state
    cdef CPotentialWrapper wrapped_potential
    cdef double[::1] _time_knots
    cdef double[::1] _c_only_params
    cdef object _param_arrays
    cdef double[:, ::1] _origin_arrays
    cdef double[:, :, ::1] _rotation_matrices
    # Keep references to prevent garbage collection of temporary arrays
    cdef object _origin_flat_arr
    cdef list _param_value_arrays

    def __init__(
        self,
        double G,
        CPotentialWrapper wrapped_potential,
        double[::1] time_knots,
        list interp_params,
        dict param_arrays,
        dict param_element_counts,
        double[::1] c_only_params,
        double[:, ::1] origin_arrays,
        double[:, :, ::1] rotation_matrices,
        str interpolation_method='cspline'
    ):
        """
        Initialize time-interpolated potential wrapper.

        Parameters
        ----------
        wrapped_potential : CPotentialWrapper
            The potential to wrap with time interpolation
        time_knots : array_like
            Time values for interpolation knots
        param_arrays : dict
            Dictionary mapping parameter names to arrays of values at each time knot
        param_element_counts : dict
            Dictionary mapping parameter names to number of elements per parameter
        c_only_params : array_like
            C-only parameters (e.g., nmax, lmax for SCF) that are prepended to regular params
        origin_arrays : array_like
            Array of origin vectors at each time knot, shape (n_knots, n_dim)
        rotation_matrices : array_like
            Array of rotation matrices at each time knot, shape (n_knots, n_dim, n_dim)
        interpolation_method : str, optional
            Interpolation type: 'linear', 'cubic', 'akima', or 'steffen'. Default is
            linear interpolation
        """
        if USE_GSL != 1:
            raise RuntimeError(
                "TimeInterpolatedPotential requires GSL support. Please install GSL "
                "and rebuild gala with GSL support to use this potential."
            )

        # we need to keep these references because they are not stored on the parent
        # potential class
        self.wrapped_potential = wrapped_potential
        self._time_knots = np.ascontiguousarray(time_knots, dtype=np.float64)
        self._param_arrays = {
            k: np.ascontiguousarray(v, dtype=np.float64)
            for k, v in param_arrays.items()
        }
        self._c_only_params = np.ascontiguousarray(c_only_params, dtype=np.float64)
        self._origin_arrays = np.ascontiguousarray(origin_arrays, dtype=np.float64)
        self._rotation_matrices = np.ascontiguousarray(
            rotation_matrices, dtype=np.float64
        )

        cdef:
            int n_knots = len(time_knots)
            int n_dim = 3  # required
            int n_c_only = len(c_only_params)
            # n_params is the number of TimeInterpParam objects
            # This is 1 (G) + n_c_only + len(param_arrays)
            int n_params = 1 + n_c_only + len(param_arrays)

            const gsl_interp_type *gsl_interp_type_ptr
            double[::1] time_knots_c = self._time_knots
            double[::1] c_only_params_c = self._c_only_params
            double[::1] param_values_view
            double[::1] origin_flat
            double rotation_matrix[9]  # one instance of rotation matrix
            double origin_val  # temporary for passing address of scalar origin
            int param_idx, i, j, result
            int n_elements

        # Map interpolation
        if interpolation_method == 'linear':
            gsl_interp_type_ptr = gsl_interp_linear
        elif interpolation_method == 'cspline':
            gsl_interp_type_ptr = gsl_interp_cspline
        elif interpolation_method == 'akima':
            gsl_interp_type_ptr = gsl_interp_akima
        elif interpolation_method == 'steffen':
            gsl_interp_type_ptr = gsl_interp_steffen
        else:
            msg = f"Unknown interpolation method: {interpolation_method}"
            raise ValueError(msg)

        if USE_GSL == 1:
            # interpolation state maintains parameter, rotation matrix, and origin
            # interpolators
            self.interp_state = time_interp_alloc(n_params, n_dim, gsl_interp_type_ptr)
            if self.interp_state == NULL:
                msg = "Failed to allocate interpolation state"
                raise MemoryError(msg)

            # Set time bounds - class enforces monotonic increasing knots
            self.interp_state.t_min = time_knots[0]
            self.interp_state.t_max = time_knots[n_knots - 1]

            # By convention in Gala, G is always at index 0, and c_only parameters
            # (e.g., nmax, lmax for SCF) follow, then the potential parameters in
            # order as defined on each class. Index starts at 1 because G is at index 0

            # Initialize G (index 0) - always constant
            # G is passed as a parameter to __init__
            result = time_interp_init_constant_param(
                &self.interp_state.params[0], &G, 1
            )
            if result != 0:
                raise RuntimeError("Failed to initialize G parameter")

            # Initialize c_only parameters (e.g., nmax, lmax for SCF) - always constant
            param_idx = 1
            for i in range(n_c_only):
                result = time_interp_init_constant_param(
                    &self.interp_state.params[param_idx], &c_only_params_c[i], 1
                )
                if result != 0:
                    raise RuntimeError(f"Failed to initialize c_only parameter at index {i}")
                param_idx += 1

            # Initialize regular parameters
            self._param_value_arrays = []  # Store to prevent GC
            for param_name, param_values_arr in self._param_arrays.items():
                param_values_view_arr = np.ascontiguousarray(param_values_arr, dtype=np.float64)
                self._param_value_arrays.append(param_values_view_arr)  # Keep reference
                param_values_view = param_values_view_arr
                n_elements = param_element_counts.get(param_name, 1)

                if param_name not in interp_params:  # constant parameter
                    # For constant parameters, pass pointer to first element
                    result = time_interp_init_constant_param(
                        &self.interp_state.params[param_idx], &param_values_view[0], n_elements
                    )

                else:  # time-interpolated parameter
                    result = time_interp_init_param(
                        &self.interp_state.params[param_idx],
                        &time_knots_c[0], &param_values_view[0], n_knots, n_elements, gsl_interp_type_ptr
                    )

                if result != 0:
                    raise RuntimeError(f"Failed to initialize parameter {param_name}")

                param_idx += 1

            # Initialize origin as a single multi-element parameter (n_elements = n_dim)
            # This always comes in as a 2D array. If it's constant, axis=0 has length 1.

            if self._origin_arrays.shape[0] == 1:  # constant
                # Flatten the constant origin - store to prevent GC
                self._origin_flat_arr = np.ascontiguousarray(self._origin_arrays[0, :], dtype=np.float64)
                origin_flat = self._origin_flat_arr  # Get memoryview
                result = time_interp_init_constant_param(
                    &self.interp_state.origin, &origin_flat[0], n_dim
                )

            elif self._origin_arrays.shape[0] == n_knots:  # time-interpolated
                # Flatten origin arrays from (n_knots, n_dim) to 1D row-major - store to prevent GC
                self._origin_flat_arr = np.ascontiguousarray(np.asarray(self._origin_arrays).ravel(), dtype=np.float64)
                origin_flat = self._origin_flat_arr  # Get memoryview
                result = time_interp_init_param(
                    &self.interp_state.origin,
                    &time_knots_c[0], &origin_flat[0], n_knots, n_dim,
                    gsl_interp_type_ptr
                )
            else:
                msg = (
                    f"Origin array has wrong shape: expected "
                    f"({1 if origin_arrays.shape[0] == 1 else n_knots}, {n_dim})"
                )
                raise ValueError(msg)

            if result != 0:
                raise RuntimeError(f"Failed to initialize origin")

            # Initialize rotation matrix interpolators
            # This always comes in as a 3D array. If it's constant, axis=0 has length 1.
            result = 0

            if self._rotation_matrices.shape[0] == 1:  # constant
                for i in range(n_dim):
                    for j in range(n_dim):
                        rotation_matrix[i*n_dim + j] = self._rotation_matrices[0, i, j]
                time_interp_init_constant_rotation(
                    &self.interp_state.rotation, rotation_matrix
                )

            elif self._rotation_matrices.shape[0] == n_knots:  # time-varying
                time_interp_init_rotation(
                    &self.interp_state.rotation,
                    &time_knots_c[0], &self._rotation_matrices[0,0,0],
                    n_knots, gsl_interp_type_ptr
                )
            else:
                raise ValueError(
                    "Rotation matrices array has wrong shape "
                    f"{self._rotation_matrices.shape}"
                )

            # Pointer to the temporary wrapped potential
            self.interp_state.wrapped_potential = <void*>self.wrapped_potential.cpotential

            self.init(
                [0.0],  # value of G doesn't matter for this wrapper
                np.zeros(n_dim, dtype=np.float64),  # placeholder origin
                np.eye(n_dim, dtype=np.float64),    # placeholder rotation
                n_dim=n_dim
            )

            # Set up function pointers (only if GSL is available)
            if USE_GSL == 1:
                self.cpotential.value[0] = <energyfunc>time_interp_value
                self.cpotential.gradient[0] = <gradientfunc>time_interp_gradient
                self.cpotential.density[0] = <densityfunc>time_interp_density
                self.cpotential.hessian[0] = <hessianfunc>time_interp_hessian

            # Store interpolation state in the state pointer
            self.cpotential.state[0] = <void*>self.interp_state

    def __dealloc__(self):
        if self.interp_state != NULL:
            time_interp_free(self.interp_state)
            self.interp_state = NULL

    @property
    def time_bounds(self):
        """Get the time bounds for interpolation."""
        if self.interp_state != NULL:
            return (self.interp_state.t_min, self.interp_state.t_max)
        return None
