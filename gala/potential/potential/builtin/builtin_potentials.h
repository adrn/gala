extern double nan_density(double t, double *pars, double *q, int n_dim);
extern double nan_value(double t, double *pars, double *q, int n_dim);
extern void nan_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double null_density(double t, double *pars, double *q, int n_dim);
extern double null_value(double t, double *pars, double *q, int n_dim);
extern void null_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void null_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double henon_heiles_value(double t, double *pars, double *q, int n_dim);
extern void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double kepler_value(double t, double *pars, double *q, int n_dim);
extern double kepler_density(double t, double *pars, double *q, int n_dim);
extern void kepler_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double isochrone_value(double t, double *pars, double *q, int n_dim);
extern void isochrone_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double isochrone_density(double t, double *pars, double *q, int n_dim);
extern void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double hernquist_value(double t, double *pars, double *q, int n_dim);
extern void hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double hernquist_density(double t, double *pars, double *q, int n_dim);
extern void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double plummer_value(double t, double *pars, double *q, int n_dim);
extern void plummer_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double plummer_density(double t, double *pars, double *q, int n_dim);
extern void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double jaffe_value(double t, double *pars, double *q, int n_dim);
extern void jaffe_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double jaffe_density(double t, double *pars, double *q, int n_dim);
extern void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double powerlawcutoff_value(double t, double *pars, double *q, int n_dim);
extern void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double powerlawcutoff_density(double t, double *pars, double *q, int n_dim);
extern void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double stone_value(double t, double *pars, double *q, int n_dim);
extern void stone_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void stone_density(double t, double *pars, double *q, int n_dim);
extern void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double sphericalnfw_value(double t, double *pars, double *q, int n_dim);
extern void sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double sphericalnfw_density(double t, double *pars, double *q, int n_dim);
extern void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double flattenednfw_value(double t, double *pars, double *q, int n_dim);
extern void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double triaxialnfw_value(double t, double *pars, double *q, int n_dim);
extern void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double satoh_value(double t, double *pars, double *q, int n_dim);
extern void satoh_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double satoh_density(double t, double *pars, double *q, int n_dim);
extern void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double kuzmin_value(double t, double *pars, double *q, int n_dim);
extern void kuzmin_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double kuzmin_density(double t, double *pars, double *q, int n_dim);

extern double miyamotonagai_value(double t, double *pars, double *q, int n_dim);
extern void miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess);
extern double miyamotonagai_density(double t, double *pars, double *q, int n_dim);

extern double mn3_value(double t, double *pars, double *q, int n_dim);
extern void mn3_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void mn3_hessian(double t, double *pars, double *q, int n_dim, double *hess);
extern double mn3_density(double t, double *pars, double *q, int n_dim);

extern double leesuto_value(double t, double *pars, double *q, int n_dim);
extern void leesuto_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double leesuto_density(double t, double *pars, double *q, int n_dim);

extern double logarithmic_value(double t, double *pars, double *q, int n_dim);
extern void logarithmic_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern void logarithmic_hessian(double t, double *pars, double *q, int n_dim, double *hess);
extern double logarithmic_density(double t, double *pars, double *q, int n_dim);

extern double longmuralibar_value(double t, double *pars, double *q, int n_dim);
extern void longmuralibar_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double longmuralibar_density(double t, double *pars, double *q, int n_dim);
extern void longmuralibar_hessian(double t, double *pars, double *q, int n_dim, double *hess);

extern double burkert_value(double t, double *pars, double *q, int n_dim);
extern void burkert_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double burkert_density(double t, double *pars, double *q, int n_dim);