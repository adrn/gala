#include "extra_compile_macros.h"

#if USE_GSL == 1

#include "time_interp.h"
#include "time_interp_wrapper.h"
#include "../src/cpotential.h"
#include "src/vectorization.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

extern "C" {

// Helper function to interpolate all parameters, origin, and rotation at time t
static int time_interp_eval_all(
    TimeInterpState *interp_state, double t, int n_dim,
    double **interp_params_out, double **interp_origin_out, double **interp_rotation_out
) {
    /*
    Interpolate all state (parameters, origin, rotation) at time t.
    Returns 0 on success, -1 on failure.
    Caller is responsible for freeing the output arrays.
    */
    if (!interp_state) return -1;

    // Calculate total number of parameter elements
    int total_param_elements = 0;
    for (int i = 0; i < interp_state->n_params; i++) {
        total_param_elements += interp_state->params[i].n_elements;
    }

    // Allocate and interpolate parameters
    double *interp_params = (double*)malloc(total_param_elements * sizeof(double));
    if (!interp_params) return -1;

    int param_offset = 0;
    for (int i = 0; i < interp_state->n_params; i++) {
        int n_elem = interp_state->params[i].n_elements;
        time_interp_eval_param(&interp_state->params[i], t, &interp_params[param_offset]);

        // Check for NaN
        for (int j = 0; j < n_elem; j++) {
            if (isnan(interp_params[param_offset + j])) {
                free(interp_params);
                return -1;
            }
        }
        param_offset += n_elem;
    }

    // Allocate and interpolate origin
    double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        return -1;
    }

    for (int i = 0; i < n_dim; i++) {
        double origin_val;
        time_interp_eval_param(&interp_state->origin[i], t, &origin_val);
        interp_origin[i] = origin_val;
        if (isnan(interp_origin[i])) {
            free(interp_params);
            free(interp_origin);
            return -1;
        }
    }

    // Allocate and interpolate rotation
    double *interp_rotation = (double*)malloc(n_dim * n_dim * sizeof(double));
    if (!interp_rotation) {
        free(interp_params);
        free(interp_origin);
        return -1;
    }

    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

    // Check for NaN in rotation matrix
    for (int i = 0; i < n_dim * n_dim; i++) {
        if (isnan(interp_rotation[i])) {
            free(interp_params);
            free(interp_origin);
            free(interp_rotation);
            return -1;
        }
    }

    *interp_params_out = interp_params;
    *interp_origin_out = interp_origin;
    *interp_rotation_out = interp_rotation;
    return 0;
}

// Time-interpolated potential evaluation function
double time_interp_value(double t, double *pars, double *q, int n_dim, void *state) {
    if (!state) return NAN;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        // Extrapolation not allowed - return NAN
        return NAN;
    }

    // Get the wrapped potential from the state
    CPotential *wrapped_pot = (CPotential*)interp_state->wrapped_potential;

    // Interpolate all state at time t
    double *interp_params, *interp_origin, *interp_rotation;
    if (time_interp_eval_all(interp_state, t, n_dim,
                             &interp_params, &interp_origin, &interp_rotation) != 0) {
        return NAN;
    }

    // Transform position using existing apply_shift_rotate function
    double *q_transformed = (double*)malloc(n_dim * sizeof(double));
    if (!q_transformed) {
        free(interp_params);
        free(interp_origin);
        free(interp_rotation);
        return NAN;
    }
    apply_shift_rotate(q, interp_origin, interp_rotation, n_dim, 0, q_transformed);

    // Evaluate wrapped potential
    double result = wrapped_pot->value[0](
        t, interp_params, q_transformed, n_dim, wrapped_pot->state[0]
    );

    free(interp_params);
    free(interp_origin);
    free(interp_rotation);
    free(q_transformed);

    return result;
}

// Time-interpolated potential gradient function
void time_interp_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) {
    if (!state || !grad) return;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        // Extrapolation not allowed - set gradient to NAN
        for (size_t i = 0; i < N * n_dim; i++) {
            grad[i] = NAN;
        }
        return;
    }

    // Get the wrapped potential from the state
    CPotential *wrapped_pot = (CPotential*)interp_state->wrapped_potential;

    // Interpolate all state at time t
    double *interp_params, *interp_origin, *interp_rotation;
    if (time_interp_eval_all(interp_state, t, n_dim,
                             &interp_params, &interp_origin, &interp_rotation) != 0) {
        for (size_t i = 0; i < N * n_dim; i++) grad[i] = NAN;
        return;
    }

    // Allocate temporary arrays for transformed coordinates
    double *q_transformed = (double*)malloc(N * n_dim * sizeof(double));
    double *grad_transformed = (double*)malloc(N * n_dim * sizeof(double));
    if (!q_transformed || !grad_transformed) {
        free(interp_params);
        free(interp_origin);
        free(interp_rotation);
        free(q_transformed);
        free(grad_transformed);
        for (size_t i = 0; i < N * n_dim; i++) grad[i] = NAN;
        return;
    }
    memset(q_transformed, 0, N * n_dim * sizeof(double));
    memset(grad_transformed, 0, N * n_dim * sizeof(double));

    // Transform positions for all orbits using existing apply_shift_rotate_N function
    apply_shift_rotate_N(q, interp_origin, interp_rotation, n_dim, N, 0, q_transformed);

    // Evaluate wrapped potential gradient in transformed coordinates
    wrapped_pot->gradient[0](t, interp_params, q_transformed, n_dim, N, grad_transformed, wrapped_pot->state[0]);

    // Transform gradient back: For each orbit, apply R^T to the gradient
    // grad_out = R^T @ grad_transformed
    for (size_t orbit_idx = 0; orbit_idx < N; orbit_idx++) {
        double temp_grad[3];  // Temporary for one orbit's gradient
        for (int i = 0; i < n_dim; i++) {
            temp_grad[i] = 0.0;
            for (int j = 0; j < n_dim; j++) {
                // R^T[i,j] = R[j,i], so we use interp_rotation[j*n_dim + i]
                temp_grad[i] += interp_rotation[j*n_dim + i] * grad_transformed[orbit_idx*n_dim + j];
            }
        }
        // Copy back
        for (int i = 0; i < n_dim; i++) {
            grad[orbit_idx*n_dim + i] = temp_grad[i];
        }
    }

    free(interp_params);
    free(interp_origin);
    free(interp_rotation);
    free(q_transformed);
    free(grad_transformed);
}

// Time-interpolated potential density function
double time_interp_density(double t, double *pars, double *q, int n_dim, void *state) {
    if (!state) return NAN;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        return NAN;
    }

    // Get the wrapped potential from the state
    CPotential *wrapped_pot = (CPotential*)interp_state->wrapped_potential;

    // Interpolate all state at time t
    double *interp_params, *interp_origin, *interp_rotation;
    if (time_interp_eval_all(interp_state, t, n_dim,
                             &interp_params, &interp_origin, &interp_rotation) != 0) {
        return NAN;
    }

    // Transform position using existing apply_shift_rotate function
    double *q_transformed = (double*)malloc(n_dim * sizeof(double));
    if (!q_transformed) {
        free(interp_params);
        free(interp_origin);
        free(interp_rotation);
        return NAN;
    }
    apply_shift_rotate(q, interp_origin, interp_rotation, n_dim, 0, q_transformed);

    // Evaluate wrapped potential density
    double result = wrapped_pot->density[0](t, interp_params, q_transformed, n_dim, wrapped_pot->state[0]);

    free(interp_params);
    free(interp_origin);
    free(interp_rotation);
    free(q_transformed);

    return result;
}

// Time-interpolated potential Hessian function
void time_interp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    if (!state || !hess) return;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        // Extrapolation not allowed - set Hessian to NAN
        for (int i = 0; i < n_dim * n_dim; i++) {
            hess[i] = NAN;
        }
        return;
    }

    // Get the wrapped potential from the state
    CPotential *wrapped_pot = (CPotential*)interp_state->wrapped_potential;

    // Interpolate all state at time t
    double *interp_params, *interp_origin, *interp_rotation;
    if (time_interp_eval_all(interp_state, t, n_dim,
                             &interp_params, &interp_origin, &interp_rotation) != 0) {
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }

    // Transform position using existing apply_shift_rotate function
    double *q_transformed = (double*)malloc(n_dim * sizeof(double));
    if (!q_transformed) {
        free(interp_params);
        free(interp_origin);
        free(interp_rotation);
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }
    apply_shift_rotate(q, interp_origin, interp_rotation, n_dim, 0, q_transformed);

    // Evaluate wrapped potential Hessian in transformed coordinates
    double *hess_transformed = (double*)malloc(n_dim * n_dim * sizeof(double));
    if (!hess_transformed) {
        free(interp_params);
        free(interp_origin);
        free(interp_rotation);
        free(q_transformed);
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }
    wrapped_pot->hessian[0](t, interp_params, q_transformed, n_dim, hess_transformed, wrapped_pot->state[0]);

    // Transform Hessian back: hess = R^T @ hess_transformed @ R
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            hess[i*n_dim + j] = 0.0;
            for (int k = 0; k < n_dim; k++) {
                for (int l = 0; l < n_dim; l++) {
                    hess[i*n_dim + j] += interp_rotation[k*n_dim + i] * hess_transformed[k*n_dim + l] * interp_rotation[l*n_dim + j];
                }
            }
        }
    }

    free(interp_params);
    free(interp_origin);
    free(interp_rotation);
    free(q_transformed);
    free(hess_transformed);
}

} // extern "C"

#endif // USE_GSL == 1
