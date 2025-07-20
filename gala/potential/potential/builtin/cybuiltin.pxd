# cython: language_level=3
# cython: language=c++

cdef extern from "potential/potential/builtin/builtin_potentials.h":
    double nan_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    double nan_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void nan_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double null_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void null_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double null_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void null_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double henon_heiles_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void henon_heiles_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double kepler_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void kepler_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double kepler_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double isochrone_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void isochrone_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double isochrone_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double hernquist_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void hernquist_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double hernquist_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double plummer_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void plummer_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double plummer_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double jaffe_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void jaffe_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double jaffe_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double powerlawcutoff_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void powerlawcutoff_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double powerlawcutoff_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double stone_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void stone_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double stone_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double sphericalnfw_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void sphericalnfw_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double sphericalnfw_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double flattenednfw_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void flattenednfw_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double triaxialnfw_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void triaxialnfw_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double satoh_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void satoh_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double satoh_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double kuzmin_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void kuzmin_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double kuzmin_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double miyamotonagai_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void miyamotonagai_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil
    double miyamotonagai_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double mn3_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void mn3_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void mn3_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil
    double mn3_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double leesuto_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void leesuto_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double leesuto_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double logarithmic_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void logarithmic_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    void logarithmic_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil
    double logarithmic_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double longmuralibar_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void longmuralibar_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double longmuralibar_density(double t, double *pars, double *q, int n_dim, void *state) nogil
    void longmuralibar_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) nogil

    double burkert_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void burkert_gradient(size_t N, double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double burkert_density(double t, double *pars, double *q, int n_dim, void *state) nogil
