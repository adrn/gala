#include "time_interp.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

TimeInterpState* time_interp_alloc(int n_params, int n_dim, const gsl_interp_type *interp_type) {
    /*
    Allocate and initialize a TimeInterpState structure
    */
    TimeInterpState *state = (TimeInterpState*)calloc(1, sizeof(TimeInterpState));
    if (!state) return NULL;

    state->n_params = n_params;
    state->n_dim = n_dim;
    state->interp_type = interp_type;
    state->t_min = 0.0;
    state->t_max = 0.0;

    // Allocate parameter interpolators
    if (n_params > 0) {
        state->params = (TimeInterpParam*)calloc(n_params, sizeof(TimeInterpParam));
        if (!state->params) {
            free(state);
            return NULL;
        }
    }

    // Allocate origin interpolators
    if (n_dim > 0) {
        state->origin = (TimeInterpParam*)calloc(n_dim, sizeof(TimeInterpParam));
        if (!state->origin) {
            free(state->params);
            free(state);
            return NULL;
        }
    }

    // Initialize rotation to identity/constant
    state->rotation.is_constant = 1;
    memset(state->rotation.constant_matrix, 0, 9 * sizeof(double));
    state->rotation.constant_matrix[0] = 1.0; // Identity matrix
    state->rotation.constant_matrix[4] = 1.0;
    state->rotation.constant_matrix[8] = 1.0;

    return state;
}

void time_interp_free(TimeInterpState *state) {
    /**
    Free all allocated memory in TimeInterpState
    */
    if (!state) return;

    // Free parameter interpolators
    if (state->params) {
        for (int i = 0; i < state->n_params; i++) {
            if (state->params[i].spline) gsl_spline_free(state->params[i].spline);
            if (state->params[i].accel) gsl_interp_accel_free(state->params[i].accel);
            if (state->params[i].time_knots) free(state->params[i].time_knots);
            if (state->params[i].param_values) free(state->params[i].param_values);
        }
        free(state->params);
    }

    // Free origin interpolators
    if (state->origin) {
        for (int i = 0; i < state->n_dim; i++) {
            if (state->origin[i].spline) gsl_spline_free(state->origin[i].spline);
            if (state->origin[i].accel) gsl_interp_accel_free(state->origin[i].accel);
            if (state->origin[i].time_knots) free(state->origin[i].time_knots);
            if (state->origin[i].param_values) free(state->origin[i].param_values);
        }
        free(state->origin);
    }

    // Free rotation interpolators
    if (!state->rotation.is_constant) {
        if (state->rotation.axis_x.spline) gsl_spline_free(state->rotation.axis_x.spline);
        if (state->rotation.axis_x.accel) gsl_interp_accel_free(state->rotation.axis_x.accel);
        if (state->rotation.axis_x.time_knots) free(state->rotation.axis_x.time_knots);
        if (state->rotation.axis_x.param_values) free(state->rotation.axis_x.param_values);

        if (state->rotation.axis_y.spline) gsl_spline_free(state->rotation.axis_y.spline);
        if (state->rotation.axis_y.accel) gsl_interp_accel_free(state->rotation.axis_y.accel);
        if (state->rotation.axis_y.time_knots) free(state->rotation.axis_y.time_knots);
        if (state->rotation.axis_y.param_values) free(state->rotation.axis_y.param_values);

        if (state->rotation.axis_z.spline) gsl_spline_free(state->rotation.axis_z.spline);
        if (state->rotation.axis_z.accel) gsl_interp_accel_free(state->rotation.axis_z.accel);
        if (state->rotation.axis_z.time_knots) free(state->rotation.axis_z.time_knots);
        if (state->rotation.axis_z.param_values) free(state->rotation.axis_z.param_values);

        if (state->rotation.angle.spline) gsl_spline_free(state->rotation.angle.spline);
        if (state->rotation.angle.accel) gsl_interp_accel_free(state->rotation.angle.accel);
        if (state->rotation.angle.time_knots) free(state->rotation.angle.time_knots);
        if (state->rotation.angle.param_values) free(state->rotation.angle.param_values);
    }

    free(state);
}

int time_interp_init_param(
    TimeInterpParam *param, double *time_knots, double *values,
    int n_knots, const gsl_interp_type *interp_type
) {
    /*
    Initialize a time-varying parameter interpolator
    */
    if (!param || !time_knots || !values || n_knots < 2) return -1;

    // Check if all values are the same (effectively constant)
    int is_constant = 1;
    for (int i = 1; i < n_knots; i++) {
        if (fabs(values[i] - values[0]) > 1e-15) {
            is_constant = 0;
            break;
        }
    }

    if (is_constant || n_knots == 1) {
        return time_interp_init_constant_param(param, values[0]);
    }

    param->is_constant = 0;
    param->n_knots = n_knots;

    // Allocate and copy time knots and values
    param->time_knots = (double*)malloc(n_knots * sizeof(double));
    param->param_values = (double*)malloc(n_knots * sizeof(double));
    if (!param->time_knots || !param->param_values) {
        if (param->time_knots) free(param->time_knots);
        if (param->param_values) free(param->param_values);
        return -1;
    }

    memcpy(param->time_knots, time_knots, n_knots * sizeof(double));
    memcpy(param->param_values, values, n_knots * sizeof(double));

    // Initialize GSL interpolation objects
    param->spline = gsl_spline_alloc(interp_type, n_knots);
    param->accel = gsl_interp_accel_alloc();

    if (!param->spline || !param->accel) {
        if (param->spline) gsl_spline_free(param->spline);
        if (param->accel) gsl_interp_accel_free(param->accel);
        free(param->time_knots);
        free(param->param_values);
        return -1;
    }

    int status = gsl_spline_init(param->spline, param->time_knots, param->param_values, n_knots);
    if (status != GSL_SUCCESS) {
        gsl_spline_free(param->spline);
        gsl_interp_accel_free(param->accel);
        free(param->time_knots);
        free(param->param_values);
        return -1;
    }

    return 0;
}

int time_interp_init_constant_param(TimeInterpParam *param, double constant_value) {
    /*
    Initialize a constant parameter
    */
    if (!param) return -1;

    memset(param, 0, sizeof(TimeInterpParam));
    param->is_constant = 1;
    param->constant_value = constant_value;

    return 0;
}

int time_interp_init_rotation(
    TimeInterpRotation *rot, double *time_knots, double *matrices,
    int n_knots, const gsl_interp_type *interp_type
) {
    /*
    Initialize rotation interpolation using axis-angle representation
    */
    if (!rot || !time_knots || !matrices || n_knots < 1) return -1;

    if (n_knots == 1) {
        return time_interp_init_constant_rotation(rot, matrices);
    }

    // Check if all rotation matrices are the same
    int is_constant = 1;
    for (int i = 1; i < n_knots; i++) {
        for (int j = 0; j < 9; j++) {
            if (fabs(matrices[i*9 + j] - matrices[j]) > 1e-15) {
                is_constant = 0;
                break;
            }
        }
        if (!is_constant) break;
    }

    if (is_constant) {
        return time_interp_init_constant_rotation(rot, matrices);
    }

    rot->is_constant = 0;

    // Convert rotation matrices to axis-angle representation
    double *axis_x_vals = (double*)malloc(n_knots * sizeof(double));
    double *axis_y_vals = (double*)malloc(n_knots * sizeof(double));
    double *axis_z_vals = (double*)malloc(n_knots * sizeof(double));
    double *angle_vals = (double*)malloc(n_knots * sizeof(double));

    if (!axis_x_vals || !axis_y_vals || !axis_z_vals || !angle_vals) {
        free(axis_x_vals);
        free(axis_y_vals);
        free(axis_z_vals);
        free(angle_vals);
        return -1;
    }

    for (int i = 0; i < n_knots; i++) {
        double axis[3], angle;
        rotation_matrix_to_axis_angle(&matrices[i*9], axis, &angle);
        axis_x_vals[i] = axis[0];
        axis_y_vals[i] = axis[1];
        axis_z_vals[i] = axis[2];
        angle_vals[i] = angle;
    }

    // Initialize interpolators for each component
    int status = 0;
    status |= time_interp_init_param(&rot->axis_x, time_knots, axis_x_vals, n_knots, interp_type);
    status |= time_interp_init_param(&rot->axis_y, time_knots, axis_y_vals, n_knots, interp_type);
    status |= time_interp_init_param(&rot->axis_z, time_knots, axis_z_vals, n_knots, interp_type);
    status |= time_interp_init_param(&rot->angle, time_knots, angle_vals, n_knots, interp_type);

    free(axis_x_vals);
    free(axis_y_vals);
    free(axis_z_vals);
    free(angle_vals);

    return status;
}

int time_interp_init_constant_rotation(TimeInterpRotation *rot, double *matrix) {
    if (!rot || !matrix) return -1;

    memset(rot, 0, sizeof(TimeInterpRotation));
    rot->is_constant = 1;
    memcpy(rot->constant_matrix, matrix, 9 * sizeof(double));

    return 0;
}

double time_interp_eval_param(const TimeInterpParam *param, double t) {
    /*
    Evaluate a parameter at time t
    */
    if (!param) {
        return 0.0;
    }

    if (param->is_constant) {
        return param->constant_value;
    }

    // Add safety check for NULL spline before calling GSL
    if (!param->spline || !param->accel) {
        return NAN;
    }

    return gsl_spline_eval(param->spline, t, param->accel);
}

void time_interp_eval_rotation(const TimeInterpRotation *rot, double t, double *matrix) {
    /*
    Evaluate rotation matrix at time t
    */
    if (!rot || !matrix) return;

    if (rot->is_constant) {
        memcpy(matrix, rot->constant_matrix, 9 * sizeof(double));
        return;
    }

    double axis[3];
    axis[0] = time_interp_eval_param(&rot->axis_x, t);
    axis[1] = time_interp_eval_param(&rot->axis_y, t);
    axis[2] = time_interp_eval_param(&rot->axis_z, t);
    double angle = time_interp_eval_param(&rot->angle, t);

    axis_angle_to_rotation_matrix(axis, angle, matrix);
}

/*
Utility functions for rotation matrix <-> axis-angle conversion
*/

void rotation_matrix_to_axis_angle(const double *matrix, double *axis, double *angle) {
    /*
    Convert rotation matrix to axis-angle representation
    */
    double trace = matrix[0] + matrix[4] + matrix[8];
    *angle = acos((trace - 1.0) / 2.0);

    if (fabs(*angle) < 1e-15) {
        // Identity rotation
        axis[0] = 1.0;
        axis[1] = 0.0;
        axis[2] = 0.0;
        *angle = 0.0;
    } else if (fabs(*angle - M_PI) < 1e-15) {
        // 180 degree rotation - special case
        double xx = (matrix[0] + 1.0) / 2.0;
        double yy = (matrix[4] + 1.0) / 2.0;
        double zz = (matrix[8] + 1.0) / 2.0;
        double xy = matrix[1] / 2.0;
        double xz = matrix[2] / 2.0;
        double yz = matrix[5] / 2.0;

        if (xx > yy && xx > zz) {
            axis[0] = sqrt(xx);
            axis[1] = xy / axis[0];
            axis[2] = xz / axis[0];
        } else if (yy > zz) {
            axis[1] = sqrt(yy);
            axis[0] = xy / axis[1];
            axis[2] = yz / axis[1];
        } else {
            axis[2] = sqrt(zz);
            axis[0] = xz / axis[2];
            axis[1] = yz / axis[2];
        }
    } else {
        // General case
        double sin_angle = sin(*angle);
        axis[0] = (matrix[7] - matrix[5]) / (2.0 * sin_angle);
        axis[1] = (matrix[2] - matrix[6]) / (2.0 * sin_angle);
        axis[2] = (matrix[3] - matrix[1]) / (2.0 * sin_angle);

        // Normalize axis
        double norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
        if (norm > 1e-15) {
            axis[0] /= norm;
            axis[1] /= norm;
            axis[2] /= norm;
        }
    }
}

void axis_angle_to_rotation_matrix(const double *axis, double angle, double *matrix) {
    /*
    Convert axis-angle representation to rotation matrix
    */
    double c = cos(angle);
    double s = sin(angle);
    double C = 1.0 - c;
    double x = axis[0], y = axis[1], z = axis[2];

    matrix[0] = x*x*C + c;
    matrix[1] = x*y*C - z*s;
    matrix[2] = x*z*C + y*s;
    matrix[3] = y*x*C + z*s;
    matrix[4] = y*y*C + c;
    matrix[5] = y*z*C - x*s;
    matrix[6] = z*x*C - y*s;
    matrix[7] = z*y*C + x*s;
    matrix[8] = z*z*C + c;
}

int time_interp_check_bounds(const TimeInterpState *state, double t) {
    /*
    Check if time t is within bounds defined by the interpolation state
    */
    if (!state) return -1;

    if (t < state->t_min || t > state->t_max) {
        return -1; // Out of bounds
    }

    return 0; // Within bounds
}
