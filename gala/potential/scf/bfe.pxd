# cython: language_level=3
# cython: language=c++

cdef extern from "scf/src/bfe.h":
    void scf_density_helper(double *xyz, int K, double M, double r_s,
                            double *Snlm, double *Tnlm,
                            int nmax, int lmax, double *dens) nogil
    void scf_potential_helper(double *xyz, int K, double G, double M, double r_s,
                              double *Snlm, double *Tnlm,
                              int nmax, int lmax, double *potv) nogil
    void scf_gradient_helper(double *xyz, int K, double G, double M, double r_s,
                             double *Snlm, double *Tnlm,
                             int nmax, int lmax, double *grad) nogil

    double scf_value(double t, double *pars, double *q, int n_dim) nogil
    double scf_density(double t, double *pars, double *q, int n_dim) nogil
    void scf_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double scf_interp_value(double t, double *pars, double *q, int n_dim) nogil
    double scf_interp_density(double t, double *pars, double *q, int n_dim) nogil
    void scf_interp_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
