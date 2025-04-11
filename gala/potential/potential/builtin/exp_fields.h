#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern double exp_value(double t, double *pars, double *q, int n_dim);
extern void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double exp_density(double t, double *pars, double *q, int n_dim);
extern void exp_hessian(double t, double *pars, double *q, int n_dim, double *hess);

#ifdef __cplusplus
}
#endif
