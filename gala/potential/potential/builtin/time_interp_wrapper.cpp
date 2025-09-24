#include "time_interp.h"
#include "time_interp_wrapper.h"
#include "../src/cpotential.h"
#include "../../src/vectorization.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Forward declarations from cpotential.cpp
void apply_shift_rotate_N(const double *q_in, const double *q0, const double *R, int n_dim, size_t N,
                        int transpose, double *q_out);
void apply_rotate_T(double6ptr q, const double *R, int n_dim, int transpose);

extern "C" {

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

    // Interpolate parameters at time t
    double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) return NAN;
    memset(interp_params, 0, interp_state->n_params * sizeof(double));

    // TODO: note - G is wrapped up in this, but is treated as constant so should be fine.
    // G should maybe be a separate (special) parameter...
    for (int i=0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
    double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        return NAN;
    }
    memset(interp_origin, 0, n_dim * sizeof(double));

    for (int i=0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
    double *interp_rotation = (double*)malloc(n_dim * n_dim * sizeof(double));
    if (!interp_rotation) {
        free(interp_params);
        free(interp_origin);
        return NAN;
    }
    memset(interp_rotation, 0, n_dim * n_dim * sizeof(double));
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

    // Transform position using existing apply_shift_rotate function
    double q_transformed[3] = {0, 0, 0};
    apply_shift_rotate(q, interp_origin, interp_rotation, n_dim, 0, q_transformed);

    // Evaluate wrapped potential
    double result = wrapped_pot->value[0](
        t, interp_params, q_transformed, n_dim, wrapped_pot->state[0]
    );

    free(interp_params);
    free(interp_origin);
    free(interp_rotation);

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

    // Interpolate parameters at time t
    double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) {
        for (size_t i = 0; i < N * n_dim; i++) grad[i] = NAN;
        return;
    }
    memset(interp_params, 0, interp_state->n_params * sizeof(double));

    // TODO: note - G is wrapped up in this, but is treated as constant so should be fine.
    // G should maybe be a separate (special) parameter...
    for (int i = 0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
    double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        for (size_t i = 0; i < N * n_dim; i++) grad[i] = NAN;
        return;
    }
    memset(interp_origin, 0, n_dim * sizeof(double));

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
    double *interp_rotation = (double*)malloc(n_dim * n_dim * sizeof(double));
    if (!interp_rotation) {
        free(interp_params);
        free(interp_origin);
        for (size_t i = 0; i < N * n_dim; i++) grad[i] = NAN;
        return;
    }
    memset(interp_rotation, 0, n_dim * n_dim * sizeof(double));
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

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

    // Transform positions for all orbits using existing apply_shift_rotate_N function
    apply_shift_rotate_N(q, interp_origin, interp_rotation, n_dim, N, 0, q_transformed);

    // Evaluate wrapped potential gradient in transformed coordinates
    wrapped_pot->gradient[0](t, interp_params, q_transformed, n_dim, N, grad_transformed, wrapped_pot->state[0]);

    // Transform gradient back for all orbits using apply_rotate_T: grad = R^T @ grad_transformed
    for (size_t i = 0; i < N; i++) {
        apply_rotate_T(
            double6ptr{grad_transformed + i, N},
            interp_rotation,
            n_dim,
            1
        );
    }

    // Copy the transformed gradients to output
    for (size_t i = 0; i < N * n_dim; i++) {
        grad[i] = grad_transformed[i];
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

    // Interpolate parameters at time t
        double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) return NAN;
        memset(interp_params, 0, interp_state->n_params * sizeof(double));

    // TODO: note - G is wrapped up in this, but is treated as constant so should be fine.
    // G should maybe be a separate (special) parameter...
    for (int i = 0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
        double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        return NAN;
    }
        memset(interp_origin, 0, n_dim * sizeof(double));

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
        double *interp_rotation = (double*)malloc(n_dim * n_dim * sizeof(double));
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);
    if (!interp_rotation) {
        free(interp_params);
        free(interp_origin);
        return NAN;
    }
        memset(interp_rotation, 0, n_dim * n_dim * sizeof(double));

    // Transform position using existing apply_shift_rotate function
        double q_transformed[n_dim];
    apply_shift_rotate(q, interp_origin, interp_rotation, n_dim, 0, q_transformed);

    // Evaluate wrapped potential density
    double result = wrapped_pot->density[0](t, interp_params, q_transformed, n_dim, wrapped_pot->state[0]);

    free(interp_params);
    free(interp_origin);
    free(interp_rotation);

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

    // Interpolate parameters at time t
        double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) {
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }
        memset(interp_params, 0, interp_state->n_params * sizeof(double));

    // TODO: note - G is wrapped up in this, but is treated as constant so should be fine.
    // G should maybe be a separate (special) parameter...
    for (int i = 0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
        double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }
        memset(interp_origin, 0, n_dim * sizeof(double));

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
        double *interp_rotation = (double*)malloc(n_dim * n_dim * sizeof(double));
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);
    if (!interp_rotation) {
        free(interp_params);
        free(interp_origin);
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }
        memset(interp_rotation, 0, n_dim * n_dim * sizeof(double));

    // Transform position using existing apply_shift_rotate function
        double q_transformed[n_dim];
    apply_shift_rotate(q, interp_origin, interp_rotation, n_dim, 0, q_transformed);

    // Evaluate wrapped potential Hessian in transformed coordinates
    double hess_transformed[n_dim * n_dim];
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
}

} // extern "C"
