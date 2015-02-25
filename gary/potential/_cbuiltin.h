extern double hernquist_value(double *pars, double *q);
extern void hernquist_gradient(double *pars, double *q, double *grad);

extern double plummer_value(double *pars, double *q);
extern void plummer_gradient(double *pars, double *q, double *grad);

extern double jaffe_value(double *pars, double *q);
extern void jaffe_gradient(double *pars, double *q, double *grad);

extern double stone_value(double *pars, double *q);
extern void stone_gradient(double *pars, double *q, double *grad);

extern double sphericalnfw_value(double *pars, double *q);
extern void sphericalnfw_gradient(double *pars, double *q, double *grad);

extern double miyamotonagai_value(double *pars, double *q);
extern void miyamotonagai_gradient(double *pars, double *q, double *grad);

extern double leesuto_value(double *pars, double *q);
extern void leesuto_gradient(double *pars, double *q, double *grad);

extern double logarithmic_value(double *pars, double *q);
extern void logarithmic_gradient(double *pars, double *q, double *grad);
