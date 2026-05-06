#pragma once

#include "extra_compile_macros.h"

#if USE_GSL == 1
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_errno.h>
#else
// Forward declarations of GSL types when GSL is not available
typedef struct gsl_spline gsl_spline;
typedef struct gsl_interp_accel gsl_interp_accel;
typedef struct gsl_interp_type gsl_interp_type;

// Dummy extern declarations for GSL interpolation types
// These allow the code to compile but will never be used due to runtime checks
extern gsl_interp_type* gsl_interp_linear;
extern gsl_interp_type* gsl_interp_cspline;
extern gsl_interp_type* gsl_interp_akima;
extern gsl_interp_type* gsl_interp_steffen;
#endif

#include <memory>
#include <vector>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

// Structure to hold interpolation state for a single parameter
// Supports both scalar (n_elements=1) and multi-element array parameters
typedef struct {
    gsl_spline **splines;       // Array of splines (one per element)
    gsl_interp_accel **accels;  // Array of accelerators (one per element)
    double *time_knots;         // Shared time knots for all elements
    double **param_values;      // param_values[element][time_knot]
    int n_knots;
    int n_elements;             // Number of elements in this parameter
    int is_constant;
    double *constant_values;    // Array of constant values (length n_elements)
} TimeInterpParam;

// Structure to hold rotation interpolation state using axis-angle representation
typedef struct {
    TimeInterpParam axis_x;  // x-component of rotation axis
    TimeInterpParam axis_y;  // y-component of rotation axis
    TimeInterpParam axis_z;  // z-component of rotation axis
    TimeInterpParam angle;   // rotation angle
    int is_constant;
    double constant_matrix[9];  // flattened 3x3 rotation matrix for constant case
} TimeInterpRotation;

// Global state structure for time interpolation of potential parameters
typedef struct {
    TimeInterpParam *params;      // Array of parameter interpolators
    TimeInterpParam origin;       // Origin interpolator (n_elements = n_dim)
    TimeInterpRotation rotation;  // Rotation interpolator
    void *wrapped_potential;      // Pointer to the wrapped CPotential
    int n_params;
    int n_dim;
    const gsl_interp_type *interp_type; // interpolation type (linear, cubic, etc.)
    double t_min;  // minimum time for bounds checking
    double t_max;  // maximum time for bounds checking
} TimeInterpState;

#if USE_GSL == 1
// Function prototypes (GSL available)
TimeInterpState* time_interp_alloc(int n_params, int n_dim, const gsl_interp_type *interp_type);
void time_interp_free(TimeInterpState *state);

int time_interp_init_param(TimeInterpParam *param, double *time_knots, double *values,
                          int n_knots, int n_elements, const gsl_interp_type *interp_type);
int time_interp_init_constant_param(TimeInterpParam *param, double *constant_values, int n_elements);

int time_interp_init_rotation(TimeInterpRotation *rot, double *time_knots, double *matrices,
                             int n_knots, const gsl_interp_type *interp_type);
int time_interp_init_constant_rotation(TimeInterpRotation *rot, double *matrix);

void time_interp_eval_param(const TimeInterpParam *param, double t, double *output_values);
void time_interp_eval_rotation(const TimeInterpRotation *rot, double t, double *matrix);

// Utility functions for rotation matrix <-> axis-angle conversion
void rotation_matrix_to_axis_angle(const double *matrix, double *axis, double *angle);
void axis_angle_to_rotation_matrix(const double *axis, double angle, double *matrix);

// Bounds checking
int time_interp_check_bounds(const TimeInterpState *state, double t);
#else
// Dummy implementations when GSL is not available
static inline TimeInterpState* time_interp_alloc(int n_params, int n_dim, const gsl_interp_type *interp_type) { return NULL; }
static inline void time_interp_free(TimeInterpState *state) {}
static inline int time_interp_init_param(TimeInterpParam *param, double *time_knots, double *values,
                          int n_knots, int n_elements, const gsl_interp_type *interp_type) { return -1; }
static inline int time_interp_init_constant_param(TimeInterpParam *param, double *constant_values, int n_elements) { return -1; }
static inline int time_interp_init_rotation(TimeInterpRotation *rot, double *time_knots, double *matrices,
                             int n_knots, const gsl_interp_type *interp_type) { return -1; }
static inline int time_interp_init_constant_rotation(TimeInterpRotation *rot, double *matrix) { return -1; }
static inline void time_interp_eval_param(const TimeInterpParam *param, double t, double *output_values) {}
static inline void time_interp_eval_rotation(const TimeInterpRotation *rot, double t, double *matrix) {}
static inline void rotation_matrix_to_axis_angle(const double *matrix, double *axis, double *angle) {}
static inline void axis_angle_to_rotation_matrix(const double *axis, double angle, double *matrix) {}
static inline int time_interp_check_bounds(const TimeInterpState *state, double t) { return -1; }
#endif // USE_GSL == 1

#ifdef __cplusplus
}
#endif
