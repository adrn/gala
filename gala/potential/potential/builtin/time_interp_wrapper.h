#ifndef TIME_INTERP_WRAPPER_H
#define TIME_INTERP_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes for time-interpolated potential evaluation
double time_interp_value(double t, double *pars, double *q, int n_dim, void *state);
void time_interp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
double time_interp_density(double t, double *pars, double *q, int n_dim, void *state);
void time_interp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

#ifdef __cplusplus
}
#endif

#endif // TIME_INTERP_WRAPPER_H
