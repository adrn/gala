#include "time_interp.h"
#include "time_interp_wrapper.h"
#include "../src/cpotential.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

extern "C" {

// Time-interpolated potential evaluation function
double time_interp_value(double t, double *pars, double *q, int n_dim, void *state) {
    if (!state) return NAN;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        // Extrapolation not allowed - raise error by returning NAN
        return NAN;
    }

    // Get the wrapped potential from the first parameter
    CPotential *wrapped_pot = (CPotential*)((void**)pars)[0];

    // Interpolate parameters at time t
    double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) return NAN;

    for (int i = 0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
    double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        return NAN;
    }

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
    double interp_rotation[9];
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

    // Transform position: q_transformed = R @ (q - origin)
    double q_transformed[n_dim];
    for (int i = 0; i < n_dim; i++) {
        q_transformed[i] = 0.0;
        for (int j = 0; j < n_dim; j++) {
            q_transformed[i] += interp_rotation[i*n_dim + j] * (q[j] - interp_origin[j]);
        }
    }

    // Update wrapped potential parameters, origin, and rotation
    memcpy(wrapped_pot->parameters[0], interp_params, interp_state->n_params * sizeof(double));
    memcpy(wrapped_pot->q0[0], interp_origin, n_dim * sizeof(double));
    memcpy(wrapped_pot->R[0], interp_rotation, n_dim * n_dim * sizeof(double));

    // Evaluate wrapped potential
    double result = wrapped_pot->value[0](t, wrapped_pot->parameters[0], q_transformed, n_dim, wrapped_pot->state[0]);

    free(interp_params);
    free(interp_origin);

    return result;
}

// Time-interpolated potential gradient function
void time_interp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state) {
    if (!state || !grad) return;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        // Extrapolation not allowed - set gradient to NAN
        for (int i = 0; i < n_dim; i++) {
            grad[i] = NAN;
        }
        return;
    }

    // Get the wrapped potential from the first parameter
    CPotential *wrapped_pot = (CPotential*)((void**)pars)[0];

    // Interpolate parameters at time t
    double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) {
        for (int i = 0; i < n_dim; i++) grad[i] = NAN;
        return;
    }

    for (int i = 0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
    double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        for (int i = 0; i < n_dim; i++) grad[i] = NAN;
        return;
    }

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
    double interp_rotation[9];
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

    // Transform position: q_transformed = R @ (q - origin)
    double q_transformed[n_dim];
    for (int i = 0; i < n_dim; i++) {
        q_transformed[i] = 0.0;
        for (int j = 0; j < n_dim; j++) {
            q_transformed[i] += interp_rotation[i*n_dim + j] * (q[j] - interp_origin[j]);
        }
    }

    // Update wrapped potential parameters, origin, and rotation
    memcpy(wrapped_pot->parameters[0], interp_params, interp_state->n_params * sizeof(double));
    memcpy(wrapped_pot->q0[0], interp_origin, n_dim * sizeof(double));
    memcpy(wrapped_pot->R[0], interp_rotation, n_dim * n_dim * sizeof(double));

    // Evaluate wrapped potential gradient in transformed coordinates
    double grad_transformed[n_dim];
    wrapped_pot->gradient[0](t, wrapped_pot->parameters[0], q_transformed, n_dim, grad_transformed, wrapped_pot->state[0]);

    // Transform gradient back: grad = R^T @ grad_transformed
    for (int i = 0; i < n_dim; i++) {
        grad[i] = 0.0;
        for (int j = 0; j < n_dim; j++) {
            grad[i] += interp_rotation[j*n_dim + i] * grad_transformed[j]; // R^T
        }
    }

    free(interp_params);
    free(interp_origin);
}

// Time-interpolated potential density function
double time_interp_density(double t, double *pars, double *q, int n_dim, void *state) {
    if (!state) return NAN;

    TimeInterpState *interp_state = (TimeInterpState*)state;

    // Check time bounds
    if (time_interp_check_bounds(interp_state, t) != 0) {
        return NAN;
    }

    // Get the wrapped potential from the first parameter
    CPotential *wrapped_pot = (CPotential*)((void**)pars)[0];

    // Interpolate parameters at time t
    double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) return NAN;

    for (int i = 0; i < interp_state->n_params; i++) {
        interp_params[i] = time_interp_eval_param(&interp_state->params[i], t);
    }

    // Interpolate origin
    double *interp_origin = (double*)malloc(n_dim * sizeof(double));
    if (!interp_origin) {
        free(interp_params);
        return NAN;
    }

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
    double interp_rotation[9];
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

    // Transform position: q_transformed = R @ (q - origin)
    double q_transformed[n_dim];
    for (int i = 0; i < n_dim; i++) {
        q_transformed[i] = 0.0;
        for (int j = 0; j < n_dim; j++) {
            q_transformed[i] += interp_rotation[i*n_dim + j] * (q[j] - interp_origin[j]);
        }
    }

    // Update wrapped potential parameters, origin, and rotation
    memcpy(wrapped_pot->parameters[0], interp_params, interp_state->n_params * sizeof(double));
    memcpy(wrapped_pot->q0[0], interp_origin, n_dim * sizeof(double));
    memcpy(wrapped_pot->R[0], interp_rotation, n_dim * n_dim * sizeof(double));

    // Evaluate wrapped potential density
    double result = wrapped_pot->density[0](t, wrapped_pot->parameters[0], q_transformed, n_dim, wrapped_pot->state[0]);

    free(interp_params);
    free(interp_origin);

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

    // Get the wrapped potential from the first parameter
    CPotential *wrapped_pot = (CPotential*)((void**)pars)[0];

    // Interpolate parameters at time t
    double *interp_params = (double*)malloc(interp_state->n_params * sizeof(double));
    if (!interp_params) {
        for (int i = 0; i < n_dim * n_dim; i++) hess[i] = NAN;
        return;
    }

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

    for (int i = 0; i < n_dim; i++) {
        interp_origin[i] = time_interp_eval_param(&interp_state->origin[i], t);
    }

    // Interpolate rotation matrix
    double interp_rotation[9];
    time_interp_eval_rotation(&interp_state->rotation, t, interp_rotation);

    // Transform position: q_transformed = R @ (q - origin)
    double q_transformed[n_dim];
    for (int i = 0; i < n_dim; i++) {
        q_transformed[i] = 0.0;
        for (int j = 0; j < n_dim; j++) {
            q_transformed[i] += interp_rotation[i*n_dim + j] * (q[j] - interp_origin[j]);
        }
    }

    // Update wrapped potential parameters, origin, and rotation
    memcpy(wrapped_pot->parameters[0], interp_params, interp_state->n_params * sizeof(double));
    memcpy(wrapped_pot->q0[0], interp_origin, n_dim * sizeof(double));
    memcpy(wrapped_pot->R[0], interp_rotation, n_dim * n_dim * sizeof(double));

    // Evaluate wrapped potential Hessian in transformed coordinates
    double hess_transformed[n_dim * n_dim];
    wrapped_pot->hessian[0](t, wrapped_pot->parameters[0], q_transformed, n_dim, hess_transformed, wrapped_pot->state[0]);

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
}

} // extern "C"
