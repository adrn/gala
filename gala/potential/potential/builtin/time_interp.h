#pragma once

#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_errno.h>
#include <memory>
#include <vector>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

// Structure to hold interpolation state for a single parameter
typedef struct {
    gsl_spline *spline;
    gsl_interp_accel *accel;
    double *time_knots;
    double *param_values;
    int n_knots;
    int is_constant;
    double constant_value;
} TimeInterpParam;

// Structure to hold rotation interpolation state using axis-angle representation
typedef struct {
    TimeInterpParam axis_x;  // x-component of rotation axis
    TimeInterpParam axis_y;  // y-component of rotation axis
    TimeInterpParam axis_z;  // z-component of rotation axis
    TimeInterpParam angle;   // rotation angle
    int is_constant;
    double constant_matrix[9]; // flattened 3x3 rotation matrix for constant case
} TimeInterpRotation;

// Main state structure for time interpolation
typedef struct {
    TimeInterpParam *params;      // Array of parameter interpolators
    TimeInterpParam *origin;      // Array for origin components (ndim)
    TimeInterpRotation rotation;  // Rotation interpolator
    void *wrapped_potential;      // Pointer to the wrapped CPotential
    int n_params;
    int n_dim;
    const gsl_interp_type *interp_type; // interpolation type (linear, cubic, etc.)
    double t_min;  // minimum time for bounds checking
    double t_max;  // maximum time for bounds checking
} TimeInterpState;

// Function prototypes
TimeInterpState* time_interp_alloc(int n_params, int n_dim, const gsl_interp_type *interp_type);
void time_interp_free(TimeInterpState *state);

int time_interp_init_param(TimeInterpParam *param, double *time_knots, double *values,
                          int n_knots, const gsl_interp_type *interp_type);
int time_interp_init_constant_param(TimeInterpParam *param, double constant_value);

int time_interp_init_rotation(TimeInterpRotation *rot, double *time_knots, double *matrices,
                             int n_knots, const gsl_interp_type *interp_type);
int time_interp_init_constant_rotation(TimeInterpRotation *rot, double *matrix);

double time_interp_eval_param(const TimeInterpParam *param, double t);
void time_interp_eval_rotation(const TimeInterpRotation *rot, double t, double *matrix);

// Utility functions for rotation matrix <-> axis-angle conversion
void rotation_matrix_to_axis_angle(const double *matrix, double *axis, double *angle);
void axis_angle_to_rotation_matrix(const double *axis, double angle, double *matrix);

// Bounds checking
int time_interp_check_bounds(const TimeInterpState *state, double t);

#ifdef __cplusplus
}
#endif
