extern void scf_density_helper(double *xyz, int K, double M, double r_s,
                               double *Snlm, double *Tnlm,
                               int nmax, int lmax, double *dens);

extern void scf_potential_helper(double *xyz, int K,
                                 double G, double M, double r_s,
                                 double *Snlm, double *Tnlm,
                                 int nmax, int lmax, double *val);

extern void scf_gradient_helper(double *xyz, int K,
                                double G, double M, double r_s,
                                double *Snlm, double *Tnlm,
                                int nmax, int lmax, double *grad);

extern double scf_value(double t, double *pars, double *q, int n_dim);
extern void scf_gradient(double t, double *pars, double *q, int n_dim,
                         double *grad);
extern double scf_density(double t, double *pars, double *q, int n_dim);

extern double scf_interp_value(double t, double *pars, double *q, int n_dim);
extern void scf_interp_gradient(double t, double *pars, double *q, int n_dim,
                                double *grad);
extern double scf_interp_density(double t, double *pars, double *q, int n_dim);
