extern double mp_potential(double t, double *pars, double *q, int n_dim);
extern double mp_density(double t, double *pars, double *q, int n_dim);
extern void mp_gradient(double t, double *pars, double *q, int n_dim, double *grad);

extern double mpetd_potential(double t, double *pars, double *q, int n_dim);
extern double mpetd_density(double t, double *pars, double *q, int n_dim);
extern void mpetd_gradient(double t, double *pars, double *q, int n_dim, double *grad);

extern void mp_density_helper(double *xyz, int K,
                               double M, double r_s,
                               double *anlm, double *bnlm,
                               int lmax, int inner, double *dens);

extern void mp_potential_helper(double *xyz, int K,
                                 double G, double M, double r_s,
                                 double *anlm, double *bnlm,
                                 int lmax, int inner, double *val);

extern void mp_gradient_helper(double *xyz, int K,
                                double G, double M, double r_s,
                                double *anlm, double *bnlm,
                                int lmax, int inner, double *grad);

extern double mp_rho_lm(double r, double phi, double X, int l, int m, int inner);
extern double mp_phi_lm(double r, double phi, double X, int l, int m, int inner);
extern void mp_sph_grad_phi_lm(double r, double phi, double X, int l, int m, int lmax, int inner, double *sphgrad);


extern double axisym_cylspline_value(double t, double *pars, double *q, int n_dim);
extern void axisym_cylspline_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double axisym_cylspline_density(double t, double *pars, double *q, int n_dim);
