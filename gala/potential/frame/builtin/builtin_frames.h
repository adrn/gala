extern double static_frame_hamiltonian(double t, double *pars, double *qp);
extern void static_frame_gradient(double t, double *pars, double *qp, double *grad);
extern void static_frame_hessian(double t, double *pars, double *qp, double *hess);

extern double constant_rotating_frame_hamiltonian(double t, double *pars, double *qp);
extern void constant_rotating_frame_gradient(double t, double *pars, double *qp, double *grad);
extern void constant_rotating_frame_hessian(double t, double *pars, double *qp, double *hess);
