/*
    Implementation of the diffusion RNG helpers, Cholesky factorization, and the
    builtin C diffusion models declared in diffusion.h.

    GSL is only touched inside the RNG helpers, guarded by USE_GSL, so that the
    Cython layer never needs to see GSL headers.
*/
#include "extra_compile_macros.h"
#include <cmath>
#include <cstdlib>
#include "diffusion.h"

#if USE_GSL == 1
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#endif

extern "C" {

void *gala_diffusion_rng_alloc(unsigned long long seed) {
#if USE_GSL == 1
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, (unsigned long)seed);
    return (void *)r;
#else
    (void)seed;
    return NULL;
#endif
}

double gala_diffusion_rng_gaussian(void *rng) {
#if USE_GSL == 1
    return gsl_ran_ugaussian((gsl_rng *)rng);
#else
    (void)rng;
    return 0.0;
#endif
}

void gala_diffusion_rng_free(void *rng) {
#if USE_GSL == 1
    if (rng != NULL) {
        gsl_rng_free((gsl_rng *)rng);
    }
#else
    (void)rng;
#endif
}

int gala_diffusion_cholesky(const double *M, int n_dim, double *L) {
    // Standard lower-triangular Cholesky (row-major). Non-positive-definite
    // directions are clamped to zero so that positive-semidefinite (including
    // all-zero) inputs still produce a usable factor.
    for (int i = 0; i < n_dim * n_dim; i++) {
        L[i] = 0.0;
    }
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j <= i; j++) {
            double s = M[i * n_dim + j];
            for (int k = 0; k < j; k++) {
                s -= L[i * n_dim + k] * L[j * n_dim + k];
            }
            if (i == j) {
                L[i * n_dim + j] = (s > 0.0) ? sqrt(s) : 0.0;
            } else {
                double Ljj = L[j * n_dim + j];
                L[i * n_dim + j] = (Ljj != 0.0) ? (s / Ljj) : 0.0;
            }
        }
    }
    return 0;
}

// ---- builtin diffusion models ----

void constant_diag_diffusion(double t, double *pars, double *q, double *v,
                             int n_dim, double *drift, double *M, void *state) {
    (void)t;
    (void)q;
    (void)v;
    (void)state;
    for (int i = 0; i < n_dim; i++) {
        drift[i] = 0.0;
    }
    for (int i = 0; i < n_dim * n_dim; i++) {
        M[i] = 0.0;
    }
    // returns_factor = 1: fill the factor B = diag(sqrt(D_i))
    for (int i = 0; i < n_dim; i++) {
        double Di = pars[i];
        M[i * n_dim + i] = (Di > 0.0) ? sqrt(Di) : 0.0;
    }
}

void constant_tensor_diffusion(double t, double *pars, double *q, double *v,
                               int n_dim, double *drift, double *M, void *state) {
    (void)t;
    (void)q;
    (void)v;
    (void)state;
    for (int i = 0; i < n_dim; i++) {
        drift[i] = 0.0;
    }
    // returns_factor = 0: M is the (symmetric) diffusion tensor D itself
    for (int i = 0; i < n_dim * n_dim; i++) {
        M[i] = pars[i];
    }
}

void example_radial_diffusion(double t, double *pars, double *q, double *v,
                              int n_dim, double *drift, double *M, void *state) {
    (void)t;
    (void)v;
    (void)state;
    // pars = [D_0, ..., D_{n_dim-1}, r_s]
    double r2 = 0.0;
    for (int i = 0; i < n_dim; i++) {
        r2 += q[i] * q[i];
    }
    double r = sqrt(r2);
    double r_s = pars[n_dim];
    double scale = exp(-r / r_s); // amplitude decays with radius; TEMPLATE only

    for (int i = 0; i < n_dim; i++) {
        drift[i] = 0.0;
    }
    for (int i = 0; i < n_dim * n_dim; i++) {
        M[i] = 0.0;
    }
    // returns_factor = 1: fill factor B = diag(sqrt(D_i)) * scale
    for (int i = 0; i < n_dim; i++) {
        double Di = pars[i];
        M[i * n_dim + i] = ((Di > 0.0) ? sqrt(Di) : 0.0) * scale;
    }
}

} // extern "C"
