#ifndef TIME_INTERP_WRAPPER_H
#define TIME_INTERP_WRAPPER_H

#include <stddef.h>
#include "extra_compile_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes for time-interpolated potential evaluation
extern double time_interp_value(double t, double *pars, double *q, int n_dim, void *state);
extern void time_interp_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double time_interp_density(double t, double *pars, double *q, int n_dim, void *state);
extern void time_interp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

#ifdef __cplusplus
}
#endif

#endif // TIME_INTERP_WRAPPER_H
