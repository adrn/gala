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

# GSL interpolation types
cdef extern from "gsl/gsl_interp.h":
    ctypedef struct gsl_interp_type:
        pass

    gsl_interp_type * gsl_interp_linear
    gsl_interp_type * gsl_interp_cspline
    gsl_interp_type * gsl_interp_akima
    gsl_interp_type * gsl_interp_steffen

# Time interpolation state structure
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
        TimeInterpParam *origin
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
                              int n_knots, const gsl_interp_type *interp_type)
    int time_interp_init_constant_param(TimeInterpParam *param, double constant_value)
    int time_interp_init_rotation(TimeInterpRotation *rot, double *time_knots, double *matrices,
                                 int n_knots, const gsl_interp_type *interp_type)
    int time_interp_init_constant_rotation(TimeInterpRotation *rot, double *matrix)

cdef extern from "time_interp_wrapper.h":
    double time_interp_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void time_interp_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    double time_interp_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void time_interp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

__all__ = ['TimeInterpolatedWrapper']

cpdef check_const_array(arr, atol, rtol):
    return np.allclose(arr, arr[0], atol=atol, rtol=rtol)


cdef class TimeInterpolatedWrapper(CPotentialWrapper):
    """
    Cython wrapper for time-interpolated potentials.
    """
    cdef TimeInterpState *interp_state
    cdef CPotentialWrapper wrapped_potential
    cdef object _time_knots
    cdef object _param_arrays
    cdef object _origin_arrays
    cdef object _rotation_matrices

    def __init__(self, CPotentialWrapper wrapped_potential,
                 double[::1] time_knots,
                 dict param_arrays,
                 double[:, ::1] origin_arrays=None,
                 double[:, :, ::1] rotation_matrices=None,
                 str interp_kind='linear',
                 float const_atol=1e-15,
                 float const_rtol=0.0):
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
        origin_arrays : array_like, optional
            Array of origin vectors at each time knot, shape (n_knots, n_dim)
        rotation_matrices : array_like, optional
            Array of rotation matrices at each time knot, shape (n_knots, n_dim, n_dim)
        interp_kind : str, optional
            Interpolation type: 'linear', 'cubic', 'akima', or 'steffen'. Default is
            linear interpolation.
        const_atol : float, optional
            Absolute tolerance for constant parameter detection. Default is 1e-15.
        const_rtol : float, optional
            Relative tolerance for constant parameter detection. Default is 0.
        """
        if USE_GSL != 1:
            raise RuntimeError("GSL is required for TimeInterpolatedPotential")

        self.wrapped_potential = wrapped_potential
        self._time_knots = np.array(time_knots)
        self._param_arrays = param_arrays
        self._origin_arrays = origin_arrays
        self._rotation_matrices = rotation_matrices

        cdef:
            int n_knots = len(time_knots)
            int n_dim = wrapped_potential.cpotential.n_dim
            int n_params = wrapped_potential.cpotential.n_params[0]

            const gsl_interp_type *gsl_interp_type_ptr
            double[::1] time_knots_view
            double *time_knots_ptr
            double[::1] param_values_view
            double *param_values_ptr
            double[::1] origin_component
            double *origin_component_ptr
            double rotation_matrix[9]
            double *rotation_matrices_ptr
            double identity_matrix[9]
            int param_idx, i, j, k
            void *wrapped_pot_ptr

        if interp_kind == 'linear':
            gsl_interp_type_ptr = gsl_interp_linear
        elif interp_kind == 'cubic':
            gsl_interp_type_ptr = gsl_interp_cspline
        elif interp_kind == 'akima':
            gsl_interp_type_ptr = gsl_interp_akima
        elif interp_kind == 'steffen':
            gsl_interp_type_ptr = gsl_interp_steffen
        else:
            msg = f"Unknown interpolation kind: {interp_kind}"
            raise ValueError(msg)

        # interpolation state maintains parameter, rotation matrix, and origin
        # interpolators
        self.interp_state = time_interp_alloc(n_params, n_dim, gsl_interp_type_ptr)
        if self.interp_state == NULL:
            msg = "Failed to allocate interpolation state"
            raise MemoryError(msg)

        # Set time bounds
        self.interp_state.t_min = time_knots[0]
        self.interp_state.t_max = time_knots[n_knots - 1]

        # Initialize parameter interpolators
        time_knots_view = time_knots
        time_knots_ptr = &time_knots_view[0]

        # First initialize all parameters as constants using values from wrapped
        # potential. This ensures all parameters have valid values
        for i in range(n_params):
            param_value = wrapped_potential.cpotential.parameters[0][i]
            result = time_interp_init_constant_param(
                &self.interp_state.params[i], param_value
            )
            if result != 0:
                msg = (
                    f"Failed to initialize constant parameter {i} (value: "
                    f"{param_value})"
                )
                raise RuntimeError(msg)

        # By convention in Gala, G is always at index 0, and the potential parameters
        # follow in order as defined on each class. The order of keys in param_arrays
        # should track this order. Index starts at 1 because G is at index 0
        # TODO: what to do about integer or string or other parameter types?
        for param_idx, (param_name, param_values) in enumerate(
            param_arrays.items(), start=1
        ):
            param_values_array = np.asarray(param_values, dtype=np.float64)

            if (
                len(param_values_array) == 1 or
                check_const_array(param_values_array, const_atol, const_rtol)
            ):
                # Constant parameter, or assumed constant due to tolerance
                result = time_interp_init_constant_param(
                    &self.interp_state.params[param_idx], param_values_array[0]
                )
                if result != 0:
                    msg = f"Failed to initialize constant parameter {param_name}"
                    raise RuntimeError(msg)

            elif len(param_values_array) == n_knots:
                # Time-varying parameter to be interpolated - copy data to avoid memory
                # access issues
                param_values_copy = np.array(
                    param_values_array, dtype=np.float64, copy=True
                )
                param_values_view = param_values_copy
                param_values_ptr = &param_values_view[0]
                result = time_interp_init_param(
                    &self.interp_state.params[param_idx],
                    time_knots_ptr, param_values_ptr, n_knots, gsl_interp_type_ptr
                )
                if result != 0:
                    msg = (
                        f"Failed to initialize parameter interpolation for {param_name}"
                    )
                    raise RuntimeError(msg)

            else:
                msg = (
                    f"Parameter {param_name} has wrong length: expected 1 or {n_knots},"
                    f" got {len(param_values_array)}"
                )
                raise ValueError(msg)

        # Initialize origin interpolators
        if origin_arrays is not None:
            origin_arrays = np.asarray(origin_arrays, dtype=np.float64)
            if origin_arrays.shape[0] == 1:
                # Constant origin
                for i in range(n_dim):
                    time_interp_init_constant_param(
                        &self.interp_state.origin[i], origin_arrays[0, i]
                    )

            elif origin_arrays.shape[0] == n_knots:
                # Time-varying origin
                for i in range(n_dim):
                    origin_component = np.ascontiguousarray(origin_arrays[:, i])
                    origin_component_ptr = &origin_component[0]
                    time_interp_init_param(
                        &self.interp_state.origin[i],
                        time_knots_ptr, origin_component_ptr, n_knots,
                        gsl_interp_type_ptr
                    )
            else:
                msg = (
                    f"Origin array has wrong shape: expected "
                    f"({1 if origin_arrays.shape[0] == 1 else n_knots}, {n_dim})"
                )
                raise ValueError(msg)

        else:
            # Default to zero vector origin
            for i in range(n_dim):
                result = time_interp_init_constant_param(
                    &self.interp_state.origin[i], 0.0
                )
                if result != 0:
                    msg = f"Failed to initialize constant origin component {i}"
                    raise RuntimeError(msg)

        # Initialize rotation matrix interpolators
        if rotation_matrices is not None:
            rotation_matrices = np.asarray(rotation_matrices, dtype=np.float64)

            if rotation_matrices.shape[0] == 1:
                # Constant rotation
                for i in range(n_dim):
                    for j in range(n_dim):
                        rotation_matrix[i*n_dim + j] = rotation_matrices[0, i, j]
                time_interp_init_constant_rotation(
                    &self.interp_state.rotation, rotation_matrix
                )

            elif rotation_matrices.shape[0] == n_knots:
                # Time-varying rotation
                rotation_matrices_ptr = <double*>malloc(n_knots * n_dim * n_dim * sizeof(double))
                if rotation_matrices_ptr == NULL:
                    raise MemoryError("Failed to allocate rotation matrices array")

                for k in range(n_knots):
                    for i in range(n_dim):
                        for j in range(n_dim):
                            rotation_matrices_ptr[k*n_dim*n_dim + i*n_dim + j] = rotation_matrices[k, i, j]

                time_interp_init_rotation(
                    &self.interp_state.rotation,
                    time_knots_ptr, rotation_matrices_ptr, n_knots, gsl_interp_type_ptr
                )
                free(rotation_matrices_ptr)
            else:
                msg = (
                    f"Rotation matrices array has wrong shape: expected "
                    f"({1 if rotation_matrices.shape[0] == 1 else n_knots}, {n_dim}, "
                    f"{n_dim})"
                )
                raise ValueError(msg)
        else:
            # Default to identity rotation matrix
            for i in range(9):  # TODO: assumes n_dim=3
                identity_matrix[i] = 0.0
            identity_matrix[0] = 1.0  # Identity matrix
            identity_matrix[4] = 1.0
            identity_matrix[8] = 1.0
            time_interp_init_constant_rotation(
                &self.interp_state.rotation, identity_matrix
            )

        # Pointer to the temporary wrapped potential
        wrapped_pot_ptr = <void*>wrapped_potential.cpotential
        self.interp_state.wrapped_potential = wrapped_pot_ptr

        self.init(
            [0.0],  # value of G doesn't matter for this wrapper
            np.zeros(n_dim, dtype=np.float64),  # placeholder origin
            np.eye(n_dim, dtype=np.float64),    # placeholder rotation
            n_dim=n_dim
        )

        # Set up function pointers
        self.cpotential.value[0] = <energyfunc>time_interp_value
        self.cpotential.gradient[0] = <gradientfunc>time_interp_gradient
        self.cpotential.density[0] = <densityfunc>time_interp_density
        self.cpotential.hessian[0] = <hessianfunc>time_interp_hessian

        # Store interpolation state in the state pointer
        self.cpotential.state[0] = <void*>self.interp_state

        # TODO: is this needed? I do'nt think so...
        # Store wrapped potential pointer in parameters
        # self.cpotential.parameters[0][0] = <double><long>wrapped_pot_ptr

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
