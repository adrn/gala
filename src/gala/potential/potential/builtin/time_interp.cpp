#include "extra_compile_macros.h"

#if USE_GSL == 1

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "time_interp.h"

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
            // Free arrays of splines and accelerators
            if (state->params[i].splines) {
                for (int j = 0; j < state->params[i].n_elements; j++) {
                    if (state->params[i].splines[j]) gsl_spline_free(state->params[i].splines[j]);
                }
                free(state->params[i].splines);
            }
            if (state->params[i].accels) {
                for (int j = 0; j < state->params[i].n_elements; j++) {
                    if (state->params[i].accels[j]) gsl_interp_accel_free(state->params[i].accels[j]);
                }
                free(state->params[i].accels);
            }
            if (state->params[i].time_knots) free(state->params[i].time_knots);
            if (state->params[i].param_values) {
                for (int j = 0; j < state->params[i].n_elements; j++) {
                    if (state->params[i].param_values[j]) free(state->params[i].param_values[j]);
                }
                free(state->params[i].param_values);
            }
            if (state->params[i].constant_values) free(state->params[i].constant_values);
        }
        free(state->params);
    }

    // Free origin interpolators (these are treated as scalar, n_elements=1)
    if (state->origin) {
        for (int i = 0; i < state->n_dim; i++) {
            if (state->origin[i].splines && state->origin[i].splines[0]) {
                gsl_spline_free(state->origin[i].splines[0]);
            }
            if (state->origin[i].splines) free(state->origin[i].splines);
            if (state->origin[i].accels && state->origin[i].accels[0]) {
                gsl_interp_accel_free(state->origin[i].accels[0]);
            }
            if (state->origin[i].accels) free(state->origin[i].accels);
            if (state->origin[i].time_knots) free(state->origin[i].time_knots);
            if (state->origin[i].param_values && state->origin[i].param_values[0]) {
                free(state->origin[i].param_values[0]);
            }
            if (state->origin[i].param_values) free(state->origin[i].param_values);
            if (state->origin[i].constant_values) free(state->origin[i].constant_values);
        }
        free(state->origin);
    }

    // Free rotation interpolators
    if (!state->rotation.is_constant) {
        if (state->rotation.axis_x.splines && state->rotation.axis_x.splines[0]) {
            gsl_spline_free(state->rotation.axis_x.splines[0]);
        }
        if (state->rotation.axis_x.splines) free(state->rotation.axis_x.splines);
        if (state->rotation.axis_x.accels && state->rotation.axis_x.accels[0]) {
            gsl_interp_accel_free(state->rotation.axis_x.accels[0]);
        }
        if (state->rotation.axis_x.accels) free(state->rotation.axis_x.accels);
        if (state->rotation.axis_x.time_knots) free(state->rotation.axis_x.time_knots);
        if (state->rotation.axis_x.param_values && state->rotation.axis_x.param_values[0]) {
            free(state->rotation.axis_x.param_values[0]);
        }
        if (state->rotation.axis_x.param_values) free(state->rotation.axis_x.param_values);

        if (state->rotation.axis_y.splines && state->rotation.axis_y.splines[0]) {
            gsl_spline_free(state->rotation.axis_y.splines[0]);
        }
        if (state->rotation.axis_y.splines) free(state->rotation.axis_y.splines);
        if (state->rotation.axis_y.accels && state->rotation.axis_y.accels[0]) {
            gsl_interp_accel_free(state->rotation.axis_y.accels[0]);
        }
        if (state->rotation.axis_y.accels) free(state->rotation.axis_y.accels);
        if (state->rotation.axis_y.time_knots) free(state->rotation.axis_y.time_knots);
        if (state->rotation.axis_y.param_values && state->rotation.axis_y.param_values[0]) {
            free(state->rotation.axis_y.param_values[0]);
        }
        if (state->rotation.axis_y.param_values) free(state->rotation.axis_y.param_values);

        if (state->rotation.axis_z.splines && state->rotation.axis_z.splines[0]) {
            gsl_spline_free(state->rotation.axis_z.splines[0]);
        }
        if (state->rotation.axis_z.splines) free(state->rotation.axis_z.splines);
        if (state->rotation.axis_z.accels && state->rotation.axis_z.accels[0]) {
            gsl_interp_accel_free(state->rotation.axis_z.accels[0]);
        }
        if (state->rotation.axis_z.accels) free(state->rotation.axis_z.accels);
        if (state->rotation.axis_z.time_knots) free(state->rotation.axis_z.time_knots);
        if (state->rotation.axis_z.param_values && state->rotation.axis_z.param_values[0]) {
            free(state->rotation.axis_z.param_values[0]);
        }
        if (state->rotation.axis_z.param_values) free(state->rotation.axis_z.param_values);

        if (state->rotation.angle.splines && state->rotation.angle.splines[0]) {
            gsl_spline_free(state->rotation.angle.splines[0]);
        }
        if (state->rotation.angle.splines) free(state->rotation.angle.splines);
        if (state->rotation.angle.accels && state->rotation.angle.accels[0]) {
            gsl_interp_accel_free(state->rotation.angle.accels[0]);
        }
        if (state->rotation.angle.accels) free(state->rotation.angle.accels);
        if (state->rotation.angle.time_knots) free(state->rotation.angle.time_knots);
        if (state->rotation.angle.param_values && state->rotation.angle.param_values[0]) {
            free(state->rotation.angle.param_values[0]);
        }
        if (state->rotation.angle.param_values) free(state->rotation.angle.param_values);
    }

    free(state);
}

int time_interp_init_param(
    TimeInterpParam *param, double *time_knots, double *values,
    int n_knots, int n_elements, const gsl_interp_type *interp_type
) {
    /*
    Initialize a time-varying parameter interpolator with support for multi-element parameters.

    Input values should be in row-major order: shape (n_knots, n_elements) flattened to 1D.
    values[0] = element 0 at time 0
    values[1] = element 1 at time 0
    ...
    values[n_elements] = element 0 at time 1
    etc.
    */
    if (!param || !time_knots || !values || n_knots < 2 || n_elements < 1) return -1;

    // Check if all values are constant (all elements, all times)
    int is_constant = 1;
    for (int elem = 0; elem < n_elements; elem++) {
        for (int t = 1; t < n_knots; t++) {
            if (fabs(values[t * n_elements + elem] - values[elem]) > 1e-15) {
                is_constant = 0;
                break;
            }
        }
        if (!is_constant) break;
    }

    if (is_constant || n_knots == 1) {
        // Extract first time step values for constant case
        double *const_vals = (double*)malloc(n_elements * sizeof(double));
        if (!const_vals) return -1;
        for (int i = 0; i < n_elements; i++) {
            const_vals[i] = values[i];  // First row
        }
        int result = time_interp_init_constant_param(param, const_vals, n_elements);
        free(const_vals);
        return result;
    }

    param->is_constant = 0;
    param->n_knots = n_knots;
    param->n_elements = n_elements;

    // Allocate arrays for multi-element support
    param->splines = (gsl_spline**)calloc(n_elements, sizeof(gsl_spline*));
    param->accels = (gsl_interp_accel**)calloc(n_elements, sizeof(gsl_interp_accel*));
    param->param_values = (double**)calloc(n_elements, sizeof(double*));

    if (!param->splines || !param->accels || !param->param_values) {
        if (param->splines) free(param->splines);
        if (param->accels) free(param->accels);
        if (param->param_values) free(param->param_values);
        return -1;
    }

    // Allocate shared time knots (same for all elements)
    param->time_knots = (double*)malloc(n_knots * sizeof(double));
    if (!param->time_knots) {
        free(param->splines);
        free(param->accels);
        free(param->param_values);
        return -1;
    }
    memcpy(param->time_knots, time_knots, n_knots * sizeof(double));

    // Initialize interpolator for each element
    for (int elem = 0; elem < n_elements; elem++) {
        // Allocate values array for this element
        param->param_values[elem] = (double*)malloc(n_knots * sizeof(double));
        if (!param->param_values[elem]) {
            // Cleanup on failure
            for (int j = 0; j < elem; j++) {
                if (param->param_values[j]) free(param->param_values[j]);
                if (param->splines[j]) gsl_spline_free(param->splines[j]);
                if (param->accels[j]) gsl_interp_accel_free(param->accels[j]);
            }
            free(param->time_knots);
            free(param->splines);
            free(param->accels);
            free(param->param_values);
            return -1;
        }

        // Extract values for this element from row-major layout
        for (int t = 0; t < n_knots; t++) {
            param->param_values[elem][t] = values[t * n_elements + elem];
        }

        // Initialize GSL spline for this element
        param->splines[elem] = gsl_spline_alloc(interp_type, n_knots);
        param->accels[elem] = gsl_interp_accel_alloc();

        if (!param->splines[elem] || !param->accels[elem]) {
            // Cleanup on failure
            if (param->splines[elem]) gsl_spline_free(param->splines[elem]);
            if (param->accels[elem]) gsl_interp_accel_free(param->accels[elem]);
            for (int j = 0; j <= elem; j++) {
                if (param->param_values[j]) free(param->param_values[j]);
                if (j < elem && param->splines[j]) gsl_spline_free(param->splines[j]);
                if (j < elem && param->accels[j]) gsl_interp_accel_free(param->accels[j]);
            }
            free(param->time_knots);
            free(param->splines);
            free(param->accels);
            free(param->param_values);
            return -1;
        }

        int status = gsl_spline_init(param->splines[elem], param->time_knots,
                                     param->param_values[elem], n_knots);
        if (status != GSL_SUCCESS) {
            // Cleanup on failure
            gsl_spline_free(param->splines[elem]);
            gsl_interp_accel_free(param->accels[elem]);
            for (int j = 0; j <= elem; j++) {
                if (param->param_values[j]) free(param->param_values[j]);
                if (j < elem && param->splines[j]) gsl_spline_free(param->splines[j]);
                if (j < elem && param->accels[j]) gsl_interp_accel_free(param->accels[j]);
            }
            free(param->time_knots);
            free(param->splines);
            free(param->accels);
            free(param->param_values);
            return -1;
        }
    }

    param->constant_values = NULL;  // Not used for time-varying
    return 0;
}

int time_interp_init_constant_param(TimeInterpParam *param, double *constant_values, int n_elements) {
    /*
    Initialize a constant parameter with support for multi-element parameters
    */
    if (!param || !constant_values || n_elements < 1) return -1;

    // Clear all fields first
    memset(param, 0, sizeof(TimeInterpParam));

    // Set the constant flag and store values
    param->is_constant = 1;
    param->n_elements = n_elements;

    param->constant_values = (double*)malloc(n_elements * sizeof(double));
    if (!param->constant_values) return -1;

    memcpy(param->constant_values, constant_values, n_elements * sizeof(double));

    // Explicitly set interpolation pointers to NULL for safety
    param->splines = NULL;
    param->accels = NULL;
    param->time_knots = NULL;
    param->param_values = NULL;
    param->n_knots = 0;

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

    // Initialize interpolators for each component (each is scalar, n_elements=1)
    int status = 0;
    status |= time_interp_init_param(&rot->axis_x, time_knots, axis_x_vals, n_knots, 1, interp_type);
    status |= time_interp_init_param(&rot->axis_y, time_knots, axis_y_vals, n_knots, 1, interp_type);
    status |= time_interp_init_param(&rot->axis_z, time_knots, axis_z_vals, n_knots, 1, interp_type);
    status |= time_interp_init_param(&rot->angle, time_knots, angle_vals, n_knots, 1, interp_type);

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

void time_interp_eval_param(const TimeInterpParam *param, double t, double *output_values) {
    /*
    Evaluate a parameter at time t for all elements.

    For constant parameters: copies constant_values to output_values
    For interpolated parameters: evaluates each element's spline and writes to output_values

    output_values must be pre-allocated with size n_elements
    */
    if (!param || !output_values) {
        return;
    }

    if (param->is_constant) {
        // Copy constant values to output
        memcpy(output_values, param->constant_values, param->n_elements * sizeof(double));
        return;
    }

    // Interpolate each element
    for (int elem = 0; elem < param->n_elements; elem++) {
        // Add safety check for NULL spline before calling GSL
        if (!param->splines[elem] || !param->accels[elem]) {
            output_values[elem] = NAN;
        } else {
            output_values[elem] = gsl_spline_eval(param->splines[elem], t, param->accels[elem]);
        }
    }
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
    double angle_val;

    // Each rotation component is scalar (n_elements=1)
    time_interp_eval_param(&rot->axis_x, t, &axis[0]);
    time_interp_eval_param(&rot->axis_y, t, &axis[1]);
    time_interp_eval_param(&rot->axis_z, t, &axis[2]);
    time_interp_eval_param(&rot->angle, t, &angle_val);

    axis_angle_to_rotation_matrix(axis, angle_val, matrix);
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

#endif  // USE_GSL
