extern double nan_density(double t, double *pars, double *q);

extern double henon_heiles_value(double t, double *pars, double *q);
extern void henon_heiles_gradient(double t, double *pars, double *q, double *grad);

extern double kepler_value(double t, double *pars, double *q);
extern void kepler_gradient(double t, double *pars, double *q, double *grad);

extern double isochrone_value(double t, double *pars, double *q);
extern void isochrone_gradient(double t, double *pars, double *q, double *grad);
extern double isochrone_density(double t, double *pars, double *q);

extern double hernquist_value(double t, double *pars, double *q);
extern void hernquist_gradient(double t, double *pars, double *q, double *grad);
extern double hernquist_density(double t, double *pars, double *q);

extern double plummer_value(double t, double *pars, double *q);
extern void plummer_gradient(double t, double *pars, double *q, double *grad);
extern double plummer_density(double t, double *pars, double *q);

extern double jaffe_value(double t, double *pars, double *q);
extern void jaffe_gradient(double t, double *pars, double *q, double *grad);
extern double jaffe_density(double t, double *pars, double *q);

extern double stone_value(double t, double *pars, double *q);
extern void stone_gradient(double t, double *pars, double *q, double *grad);
extern void stone_density(double t, double *pars, double *q);

extern double sphericalnfw_value(double t, double *pars, double *q);
extern void sphericalnfw_gradient(double t, double *pars, double *q, double *grad);
extern double sphericalnfw_density(double t, double *pars, double *q);

extern double flattenednfw_value(double t, double *pars, double *q);
extern void flattenednfw_gradient(double t, double *pars, double *q, double *grad);
extern double flattenednfw_density(double t, double *pars, double *q);

extern double miyamotonagai_value(double t, double *pars, double *q);
extern void miyamotonagai_gradient(double t, double *pars, double *q, double *grad);
extern double miyamotonagai_density(double t, double *pars, double *q);

extern double leesuto_value(double t, double *pars, double *q);
extern void leesuto_gradient(double t, double *pars, double *q, double *grad);
extern double leesuto_density(double t, double *pars, double *q);

extern double logarithmic_value(double t, double *pars, double *q);
extern void logarithmic_gradient(double t, double *pars, double *q, double *grad);

extern double rotating_logarithmic_value(double t, double *pars, double *q);
extern void rotating_logarithmic_gradient(double t, double *pars, double *q, double *grad);

extern double lm10_value(double t, double *pars, double *q);
extern void lm10_gradient(double t, double *pars, double *q, double *grad);
