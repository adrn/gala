extern double nan_density(double t, double *pars, double *q, int n_dim, void *state);
extern double nan_value(double t, double *pars, double *q, int n_dim, void *state);
extern void nan_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double null_density(double t, double *pars, double *q, int n_dim, void *state);
extern double null_value(double t, double *pars, double *q, int n_dim, void *state);
extern void null_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void null_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double henon_heiles_value(double t, double *pars, double *q, int n_dim, void *state);
extern void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double kepler_value(double t, double *pars, double *q, int n_dim, void *state);
extern double kepler_density(double t, double *pars, double *q, int n_dim, void *state);
extern void kepler_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double isochrone_value(double t, double *pars, double *q, int n_dim, void *state);
extern void isochrone_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double isochrone_density(double t, double *pars, double *q, int n_dim, void *state);
extern void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double hernquist_value(double t, double *pars, double *q, int n_dim, void *state);
extern void hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double hernquist_density(double t, double *pars, double *q, int n_dim, void *state);
extern void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double plummer_value(double t, double *pars, double *q, int n_dim, void *state);
extern void plummer_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double plummer_density(double t, double *pars, double *q, int n_dim, void *state);
extern void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double jaffe_value(double t, double *pars, double *q, int n_dim, void *state);
extern void jaffe_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double jaffe_density(double t, double *pars, double *q, int n_dim, void *state);
extern void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double powerlawcutoff_value(double t, double *pars, double *q, int n_dim, void *state);
extern void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double powerlawcutoff_density(double t, double *pars, double *q, int n_dim, void *state);
extern void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double stone_value(double t, double *pars, double *q, int n_dim, void *state);
extern void stone_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void stone_density(double t, double *pars, double *q, int n_dim, void *state);
extern void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double sphericalnfw_value(double t, double *pars, double *q, int n_dim, void *state);
extern void sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double sphericalnfw_density(double t, double *pars, double *q, int n_dim, void *state);
extern void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double flattenednfw_value(double t, double *pars, double *q, int n_dim, void *state);
extern void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double triaxialnfw_value(double t, double *pars, double *q, int n_dim, void *state);
extern void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double satoh_value(double t, double *pars, double *q, int n_dim, void *state);
extern void satoh_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double satoh_density(double t, double *pars, double *q, int n_dim, void *state);
extern void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double kuzmin_value(double t, double *pars, double *q, int n_dim, void *state);
extern void kuzmin_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double kuzmin_density(double t, double *pars, double *q, int n_dim, void *state);

extern double miyamotonagai_value(double t, double *pars, double *q, int n_dim, void *state);
extern void miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);
extern double miyamotonagai_density(double t, double *pars, double *q, int n_dim, void *state);

extern double mn3_value(double t, double *pars, double *q, int n_dim, void *state);
extern void mn3_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void mn3_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);
extern double mn3_density(double t, double *pars, double *q, int n_dim, void *state);

extern double leesuto_value(double t, double *pars, double *q, int n_dim, void *state);
extern void leesuto_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double leesuto_density(double t, double *pars, double *q, int n_dim, void *state);

extern double logarithmic_value(double t, double *pars, double *q, int n_dim, void *state);
extern void logarithmic_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern void logarithmic_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);
extern double logarithmic_density(double t, double *pars, double *q, int n_dim, void *state);

extern double longmuralibar_value(double t, double *pars, double *q, int n_dim, void *state);
extern void longmuralibar_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double longmuralibar_density(double t, double *pars, double *q, int n_dim, void *state);
extern void longmuralibar_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double burkert_value(double t, double *pars, double *q, int n_dim, void *state);
extern void burkert_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state);
extern double burkert_density(double t, double *pars, double *q, int n_dim, void *state);
