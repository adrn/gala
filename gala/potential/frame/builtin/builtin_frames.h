#include <stddef.h>

extern double static_frame_hamiltonian(double t, double *pars, double *qp, int n_dim);
extern void static_frame_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void static_frame_hessian(double t, double *pars, double *qp, int n_dim, double *hess);

extern double constant_rotating_frame_2d_hamiltonian(double t, double *pars, double *qp, int n_dim);
extern void constant_rotating_frame_2d_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void constant_rotating_frame_2d_hessian(double t, double *pars, double *qp, int n_dim, double *hess);

extern double constant_rotating_frame_3d_hamiltonian(double t, double *pars, double *qp, int n_dim);
extern void constant_rotating_frame_3d_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void constant_rotating_frame_3d_hessian(double t, double *pars, double *qp, int n_dim, double *hess);
