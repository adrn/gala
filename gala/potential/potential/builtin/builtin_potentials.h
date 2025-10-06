#include <stddef.h>

#if USE_GSL == 1
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#else
// When GSL is not available, provide dummy types
typedef struct { int dummy; } gsl_spline;
typedef struct { int dummy; } gsl_interp_accel;
typedef struct { int dummy; } gsl_interp_type;

// Provide dummy GSL interpolation type constants
static const gsl_interp_type *gsl_interp_linear = NULL;
static const gsl_interp_type *gsl_interp_polynomial = NULL;
static const gsl_interp_type *gsl_interp_cspline = NULL;
static const gsl_interp_type *gsl_interp_cspline_periodic = NULL;
static const gsl_interp_type *gsl_interp_akima = NULL;
static const gsl_interp_type *gsl_interp_akima_periodic = NULL;
static const gsl_interp_type *gsl_interp_steffen = NULL;

// Provide dummy function declarations for GSL functions
static inline gsl_interp_accel* gsl_interp_accel_alloc(void) { return NULL; }
static inline void gsl_interp_accel_free(gsl_interp_accel *acc) {}
static inline gsl_spline* gsl_spline_alloc(const gsl_interp_type *T, size_t size) { return NULL; }
static inline int gsl_spline_init(gsl_spline *spline, const double *xa, const double *ya, size_t size) { return 0; }
static inline void gsl_spline_free(gsl_spline *spline) {}
#endif

// Spherical spline interpolation state structure
// Note: We always define the full struct to keep Cython happy, even when GSL is not available
typedef struct {
    gsl_spline *spline;        // Main spline for density, mass, or potential
    gsl_interp_accel *acc;     // Accelerator for main spline
    gsl_spline *rho_r_spline;  // Spline for ρ(r) * r (used in density potential calc)
    gsl_spline *rho_r2_spline; // Spline for ρ(r) * r² (used in density gradient calc)
    gsl_interp_accel *rho_r_acc;   // Accelerator for ρ(r) * r spline
    gsl_interp_accel *rho_r2_acc;  // Accelerator for ρ(r) * r² spline
    int n_knots;
    int method;
    double *r_knots;
    double *values;
} spherical_spline_state;

extern double nan_density(double t, double *pars, double *q, int n_dim, void *state);
extern double nan_value(double t, double *pars, double *q, int n_dim, void *state);
extern void nan_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double null_density(double t, double *pars, double *q, int n_dim, void *state);
extern double null_value(double t, double *pars, double *q, int n_dim, void *state);
extern void null_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void null_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double henon_heiles_value(double t, double *pars, double *q, int n_dim, void *state);
extern void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double kepler_value(double t, double *pars, double *q, int n_dim, void *state);
extern double kepler_density(double t, double *pars, double *q, int n_dim, void *state);
extern void kepler_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double isochrone_value(double t, double *pars, double *q, int n_dim, void *state);
extern void isochrone_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double isochrone_density(double t, double *pars, double *q, int n_dim, void *state);
extern void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double hernquist_value(double t, double *pars, double *q, int n_dim, void *state);
extern void hernquist_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double hernquist_density(double t, double *pars, double *q, int n_dim, void *state);
extern void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double plummer_value(double t, double *pars, double *q, int n_dim, void *state);
extern void plummer_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double plummer_density(double t, double *pars, double *q, int n_dim, void *state);
extern void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double jaffe_value(double t, double *pars, double *q, int n_dim, void *state);
extern void jaffe_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double jaffe_density(double t, double *pars, double *q, int n_dim, void *state);
extern void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double powerlawcutoff_value(double t, double *pars, double *q, int n_dim, void *state);
extern void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double powerlawcutoff_density(double t, double *pars, double *q, int n_dim, void *state);
extern void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double stone_value(double t, double *pars, double *q, int n_dim, void *state);
extern void stone_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void stone_density(double t, double *pars, double *q, int n_dim, void *state);
extern void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double sphericalnfw_value(double t, double *pars, double *q, int n_dim, void *state);
extern void sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double sphericalnfw_density(double t, double *pars, double *q, int n_dim, void *state);
extern void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double flattenednfw_value(double t, double *pars, double *q, int n_dim, void *state);
extern void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double triaxialnfw_value(double t, double *pars, double *q, int n_dim, void *state);
extern void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double satoh_value(double t, double *pars, double *q, int n_dim, void *state);
extern void satoh_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double satoh_density(double t, double *pars, double *q, int n_dim, void *state);
extern void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double kuzmin_value(double t, double *pars, double *q, int n_dim, void *state);
extern void kuzmin_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double kuzmin_density(double t, double *pars, double *q, int n_dim, void *state);

extern double miyamotonagai_value(double t, double *pars, double *q, int n_dim, void *state);
extern void miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);
extern double miyamotonagai_density(double t, double *pars, double *q, int n_dim, void *state);

extern double mn3_value(double t, double *pars, double *q, int n_dim, void *state);
extern void mn3_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void mn3_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);
extern double mn3_density(double t, double *pars, double *q, int n_dim, void *state);

extern double leesuto_value(double t, double *pars, double *q, int n_dim, void *state);
extern void leesuto_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double leesuto_density(double t, double *pars, double *q, int n_dim, void *state);

extern double logarithmic_value(double t, double *pars, double *q, int n_dim, void *state);
extern void logarithmic_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern void logarithmic_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);
extern double logarithmic_density(double t, double *pars, double *q, int n_dim, void *state);

extern double longmuralibar_value(double t, double *pars, double *q, int n_dim, void *state);
extern void longmuralibar_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double longmuralibar_density(double t, double *pars, double *q, int n_dim, void *state);
extern void longmuralibar_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state);

extern double burkert_value(double t, double *pars, double *q, int n_dim, void *state);
extern void burkert_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double burkert_density(double t, double *pars, double *q, int n_dim, void *state);

// Spherical spline interpolated potentials
extern double spherical_spline_density_value(double t, double *pars, double *q, int n_dim, void *state);
extern void spherical_spline_density_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double spherical_spline_density_density(double t, double *pars, double *q, int n_dim, void *state);

extern double spherical_spline_mass_value(double t, double *pars, double *q, int n_dim, void *state);
extern void spherical_spline_mass_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double spherical_spline_mass_density(double t, double *pars, double *q, int n_dim, void *state);

extern double spherical_spline_potential_value(double t, double *pars, double *q, int n_dim, void *state);
extern void spherical_spline_potential_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double spherical_spline_potential_density(double t, double *pars, double *q, int n_dim, void *state);
