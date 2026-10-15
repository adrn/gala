/*
copied from https://github.com/adrn/gala/blob/main/gala/potential/scf/src/bfe.c
and https://github.com/adrn/gala/blob/main/gala/potential/scf/src/bfe_helper.c
*/
#include <stdlib.h>
#include <stdio.h>
#include "extra_compile_macros.h"
#include <math.h>
#include <string.h>
#include "src/vectorization.h"

#define SQRT_FOURPI 3.544907701811031

#if USE_GSL == 1

#include "gsl/gsl_sf_legendre.h"
#include "gsl/gsl_sf_gegenbauer.h"
#include "gsl/gsl_sf_gamma.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

/* --------------------------------------------------------------------------

    Low-level helper functions

*/

/*
    Density
*/
double mp_rho_l_outer(double r, int l) {
    return l * (l+1) * pow(r, -(l+3));
}

double mp_rho_l_inner(double r, int l) {
    return l * (l+1) * pow(r, l-2);
}

double mp_rho_lm(double r, double phi, double X, int l, int m, int inner) {
    if (inner > 0) {
        return mp_rho_l_inner(r, l) * gsl_sf_legendre_sphPlm(l, m, X);
    } else {
        return mp_rho_l_outer(r, l) * gsl_sf_legendre_sphPlm(l, m, X);
    }
}

/*
    Potential
*/
double mp_phi_l_outer(double r, int l) {
    return pow(r, -(l + 1));
}

double mp_phi_l_inner(double r, int l) {
    return pow(r, l);
}

double mp_phi_lm(double r, double phi, double X, int l, int m, int inner) {
    if (inner > 0) {
        return mp_phi_l_inner(r, l) * gsl_sf_legendre_sphPlm(l, m, X);
    } else {
        return mp_phi_l_outer(r, l) * gsl_sf_legendre_sphPlm(l, m, X);
    }
}

/*
    Gradient
*/
void mp_sph_grad_phi_lm(double r, double phi, double X, int l, int m,
                        int lmax, int inner,
                        double *__restrict__ sphgradx, double *__restrict__ sphgrady, double *__restrict__ sphgradz) {
    double A, dYlm_dtheta;
    double dPhil_dr, dPhi_dphi, dPhi_dtheta;

    // spherical coord stuff
    double sintheta = sqrt(1 - X*X);

    double Phi_l, Ylm, Plm, Pl1m;
    Ylm = gsl_sf_legendre_sphPlm(l, m, X);

    // Correct: associated Legendre polynomial -- not sphPlm!
    if (m <= l) {
        Plm = gsl_sf_legendre_Plm(l, m, X);
    } else {
        Plm = 0.;
    }

    if (inner > 0) {
        Phi_l = mp_phi_l_inner(r, l);
        dPhil_dr = l*pow(r, l-1) * Ylm;
    } else {
        Phi_l = mp_phi_l_outer(r, l);
        dPhil_dr = -(l+1) * pow(r, -l-2) * Ylm;
    }

    if (l==0) {
        dYlm_dtheta = 0.;
    } else {
        // Correct: associated Legendre polynomial -- not sphPlm!
        if (m <= (l-1)) {
            Pl1m = gsl_sf_legendre_Plm(l-1, m, X);
        } else {
            Pl1m = 0.;
        }

        if (l == m) {
            A = sqrt(2*l+1) / SQRT_FOURPI * sqrt(1. / gsl_sf_gamma(l+m+1.));
        } else {
            A = sqrt(2*l+1) / SQRT_FOURPI * sqrt(gsl_sf_gamma(l-m+1.)
                                                 / gsl_sf_gamma(l+m+1.));
        }

        // fixed at sintheta = 0
        if (sintheta != 0) {
            dYlm_dtheta = A / sintheta * (l*X*Plm - (l+m)*Pl1m);
        } else {
            dYlm_dtheta = 0;
        }
    }
    dPhi_dtheta = dYlm_dtheta * Phi_l / r;

    if (m == 0) {
        dPhi_dphi = 0.;
    } else {
        dPhi_dphi = m;
    }
    dPhi_dphi *= Ylm * Phi_l;

    if (r > 0) {
        *sphgradx = dPhil_dr;
        *sphgrady = dPhi_dtheta;
        *sphgradz = dPhi_dphi;
    } else {
        *sphgradx = 0;
        *sphgrady = 0;
        *sphgradz = 0;
    }
}

/*
    High-level functions and helpers
*/

void mp_density_helper(double *xyz, int K,
                       double M, double r_s,
                       double *Slm, double *Tlm,
                       int lmax, int inner, double *dens) {

    int i,j,k, l,m;
    double s, r, X, phi;
    double cosmphi[lmax+1], sinmphi[lmax+1];
    memset(cosmphi, 0, (lmax+1)*sizeof(double));
    memset(sinmphi, 0, (lmax+1)*sizeof(double));
    for (k=0; k<K; k++) {
        j = 3*k;
        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        s = r / r_s;
        X = xyz[j+2] / r; // = cos(theta)
        phi = atan2(xyz[j+1], xyz[j+0]);

        // precompute all cos(m phi), sin(m phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        i = 0;
        for (l=0; l < (lmax+1); l++) {
            for (m=0; m <= l; m++) {
                if ((Slm[i] == 0.) & (Tlm[i] == 0.)) {
                    i++;
                    continue;
                }
                dens[k] += mp_rho_lm(s, phi, X, l, m, inner) * (
                    Slm[i] * cosmphi[m] + Tlm[i] * sinmphi[m]
                );
                i++;
            }
        }
        dens[k] *= M / (r_s*r_s*r_s);
    }
}

void mp_potential_helper(double *xyz, int K,
                         double G, double M, double r_s,
                         double *Slm, double *Tlm,
                         int lmax, int inner, double *val) {

    int i,j,k, l,m;
    double s, r, X, phi;
    double cosmphi[lmax+1], sinmphi[lmax+1];

    for (k=0; k<K; k++) {
        j = 3*k;
        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        s = r / r_s;
        X = xyz[j+2] / r; // = cos(theta)
        phi = atan2(xyz[j+1], xyz[j+0]);

        // HACK: zero out before filling;
        val[k] = 0.;

        // precompute all cos(m phi), sin(m phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m * phi);
            sinmphi[m] = sin(m * phi);
        }

        i = 0;
        for (l=0; l < (lmax+1); l++) {
            for (m=0; m < (l+1); m++) {
                if ((Slm[i] == 0.) & (Tlm[i] == 0.)) {
                    i++;
                    continue;
                }
                val[k] += mp_phi_lm(s, phi, X, l, m, inner) * (
                    Slm[i] * cosmphi[m] + Tlm[i] * sinmphi[m]
                );
                i++;
            }
        }
        if((r==0) && inner) {val[k] = 0.;}
        val[k] *= G*M/r_s;
    }
}

void mp_gradient_helper(double6ptr q, int K,
                        double G, double M, double r_s,
                        double *__restrict__ Slm, double *__restrict__ Tlm,
                        int lmax, int inner,
                        double6ptr grad) {

    int i,k,l,m;
    double s, r, X, phi;
    double sintheta, cosphi, sinphi, tmp;
    double tmp_grad[3];
    double cosmphi[lmax+1], sinmphi[lmax+1];

    for (k=0; k<K; k++) {
        r = sqrt(q.x[k]*q.x[k] + q.y[k]*q.y[k] + q.z[k]*q.z[k]);
        s = r / r_s;
        X = q.z[k] / r; // cos(theta)
        phi = atan2(q.y[k], q.x[k]);

        sintheta = sqrt(1 - X*X);
        cosphi = cos(phi);
        sinphi = sin(phi);

        // precompute all cos(m phi), sin(m phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        // zero out
        double tmp_grad2[3] = {0., 0., 0.};

        i = 0;
        // gsl_sf_legendre_deriv_array(GSL_SF_LEGENDRE_SPHARM, lmax, X,
        //                             double result_array[], double result_deriv_array[]);
        for (l=0; l<(lmax+1); l++) {
            for (m=0; m<(l+1); m++) {
                tmp = (Slm[i]*cosmphi[m] + Tlm[i]*sinmphi[m]);
                if ((Slm[i] == 0.) & (Tlm[i] == 0.)) {
                    i++;
                    continue;
                }

                mp_sph_grad_phi_lm(s, phi, X, l, m, lmax, inner, &tmp_grad[0], &tmp_grad[1], &tmp_grad[2]);
                tmp_grad2[0] += tmp_grad[0] * tmp; // r
                tmp_grad2[1] += tmp_grad[1] * tmp; // phi??

                if (sintheta != 0) {
                    tmp_grad2[2] += tmp_grad[2] * (
                        Tlm[i]*cosmphi[m] - Slm[i]*sinmphi[m]
                    ) / (s * sintheta); // theta??
                } else {
                    tmp_grad2[2] = 0.;
                }


                i++;
            }
        }
        tmp_grad[0] = tmp_grad2[0];
        tmp_grad[1] = tmp_grad2[1];
        tmp_grad[2] = tmp_grad2[2];

        // transform to cartesian
        tmp_grad2[0] = sintheta*cosphi*tmp_grad[0] + X*cosphi*tmp_grad[1]
            - sinphi*tmp_grad[2];
        tmp_grad2[1] = sintheta*sinphi*tmp_grad[0] + X*sinphi*tmp_grad[1]
            + cosphi*tmp_grad[2];
        tmp_grad2[2] = X*tmp_grad[0] - sintheta*tmp_grad[1];

        grad.x[k] += tmp_grad2[0] * G*M/(r_s*r_s);
        grad.y[k] += tmp_grad2[1] * G*M/(r_s*r_s);
        grad.z[k] += tmp_grad2[2] * G*M/(r_s*r_s);
  }
}

double mp_potential(double t, double *pars, double *q, int n_dim) {
    /*  pars:
        - G (Gravitational constant)
        - lmax
        - num_coeff
        - inner
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */

    double G = pars[0];
    int lmax = (int)pars[1];
    int num_coeff = (int)pars[2];
    int inner = (int)pars[3];
    double M = pars[4];
    double r_s = pars[5];

    double val[1] = {0.};

    double Slm[num_coeff], Tlm[num_coeff];
    for(int i=0; i < num_coeff; i++){
        Slm[i] = pars[6 + 2*i];
        Tlm[i] = pars[7 + 2*i];
    }

    mp_potential_helper(&q[0], 1,
                        G, M, r_s,
                        &Slm[0], &Tlm[0],
                        lmax, inner, &val[0]);

    return val[0];
}

void mp_gradient(double t, double *__restrict__ pars, double *__restrict__ q, int n_dim, size_t N, double *__restrict__ grad, void *__restrict__ state) {
    /*  pars:
        - G (Gravitational constant)
        - lmax
        - num_coeff
        - inner
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */
    double G = pars[0];
    int lmax = (int)pars[1];
    int num_coeff = (int)pars[2];
    int inner = (int)pars[3];
    double M = pars[4];
    double r_s = pars[5];

    double Slm[num_coeff], Tlm[num_coeff];
    for(int i=0; i<num_coeff; i++){
        Slm[i] = pars[6 + 2*i];
        Tlm[i] = pars[7 + 2*i];
    }

    mp_gradient_helper(double6ptr{q, N}, N,
                       G, M, r_s,
                       &Slm[0], &Tlm[0],
                       lmax, inner, double6ptr{grad, N});
}

void _mp_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    // This is a helper used by axisym_cylspline_gradient, but not by mp_gradient.

    double G = pars[0];
    int lmax = (int)pars[1];
    int num_coeff = (int)pars[2];
    int inner = (int)pars[3];
    double M = pars[4];
    double r_s = pars[5];

    double Slm[num_coeff], Tlm[num_coeff];
    for(int i=0; i<num_coeff; i++){
        Slm[i] = pars[6 + 2*i];
        Tlm[i] = pars[7 + 2*i];
    }

    mp_gradient_helper(q, 1,
                       G, M, r_s,
                       &Slm[0], &Tlm[0],
                       lmax, inner, grad);
}

double mp_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
        - G (Gravitational constant)
        - lmax
        - num_coeff
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */
    // double G = pars[0];
    // int lmax = (int)pars[1];
    // int num_coeff = (int)pars[2];
    // int inner = (int)pars[3];
    // double M = pars[4];
    // double r_s = pars[5];

    double val[1] = {0.};

    /* BUG HERE, to do: work out the full density as the laplacian of the
    potential and implement that as a consistency check (should be always 0)
    until then, we set the density to 0
    */
    // mp_density_helper(&q[0], 1,
    //                    M, r_s,
    //                    &pars[4], &pars[4+num_coeff],
    //                    lmax, &val[0]);

    val[0] = 0.;
    return val[0];
}

/* --------------------------------------------------------------------------

    Time-dependent Multipole Expansion

*/
double mpetd_polynomial(double t, double *coeffs, int deg) {
    double val = 0;
    for (int k=0; k < deg; k++){
        val += coeffs[k] * pow(t, k);
    }
    return val;
}

double mpetd_potential(double t, double *pars, double *q, int n_dim) {
    /*  pars:
        - G (Gravitational constant)
        - lmax
        - num_coeff
        - deg_a (time polynomial degree); len of num_coeff
        - deg_b (time polynomial degree); len of num_coeff
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */

    double G = pars[0];
    int lmax = (int)pars[1];
    int num_coeff = (int)pars[2];

    int deg_a[num_coeff];
    int deg_b[num_coeff];
    for(int i=0; i < num_coeff; i++){
        deg_a[i] = (int)pars[3 + i];
        deg_b[i] = (int)pars[3 + num_coeff + i];
    }

    double M = pars[2*num_coeff + 3];
    double r_s = pars[2*num_coeff + 4];

    // printf("mpetd_potential: lmax: %d, M: %f, r_s: %f, deg_a[0]: %d, deg_b[0]: %d, first coeff: %f\n",
    //        lmax, M, r_s, deg_a[0], deg_b[0], pars[2*num_coeff + 5]);

    // Construct the alm and blm values:
    double alm[num_coeff], blm[num_coeff];

    double val[1] = {0.};

    int i = 0;
    int k_stride = 2*num_coeff + 5;
    for (int l=0; l<(lmax+1); l++) {
        for (int m=0; m<(l+1); m++) {
            // printf("mpetd_potential: l=%d m=%d deg_a=%d deg_b=%d alpha0=%f beta0=%f\n",
            //        l, m, deg_a[i], deg_b[i], pars[k_stride], pars[k_stride + deg_a[i]]);

            alm[i] = mpetd_polynomial(t, &pars[k_stride], deg_a[i]);
            blm[i] = mpetd_polynomial(t, &pars[k_stride + deg_a[i]], deg_b[i]);
            k_stride += deg_a[i] + deg_b[i];
            i++;
        }
    }

    // for(i=0; i<num_coeff; i++)
    //     printf("in potential.c:mpetd_potential: a, b [%d]: %f %f\n", i, alm[i], blm[i]);

    mp_potential_helper(&q[0], 1,
                        G, M, r_s,
                        &alm[0], &blm[0],
                        lmax, 1, &val[0]);

    return val[0];
}

void mpetd_gradient(double t, double *__restrict__ pars, double *__restrict__ q, int n_dim, size_t N, double *__restrict__ grad, void *__restrict__ state) {
    /*  pars:
        - G (Gravitational constant)
        - lmax
        - num_coeff
        - deg_a (time polynomial degree); len of num_coeff
        - deg_b (time polynomial degree); len of num_coeff
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */
    double G = pars[0];
    int lmax = (int)pars[1];
    int num_coeff = (int)pars[2];

    int deg_a[num_coeff];
    int deg_b[num_coeff];
    for(int i=0; i < num_coeff; i++){
        deg_a[i] = (int)pars[3 + i];
        deg_b[i] = (int)pars[3 + num_coeff + i];
    }

    double M = pars[2*num_coeff + 3];
    double r_s = pars[2*num_coeff + 4];


    // Construct the alm and blm values:
    double alm[num_coeff], blm[num_coeff];

    double val[1] = {0.};
    int i = 0;
    int k_stride = 2*num_coeff + 5;
    for (int l=0; l<(lmax+1); l++) {
        for (int m=0; m<(l+1); m++) {
            alm[i] = mpetd_polynomial(t, &pars[k_stride], deg_a[i]);
            blm[i] = mpetd_polynomial(t, &pars[k_stride + deg_a[i]], deg_b[i]);
            k_stride += deg_a[i]+deg_b[i];
            i++;
        }
    }



    mp_gradient_helper(double6ptr{q, N}, N,
                        G, M, r_s,
                        &alm[0], &blm[0],
                        lmax, 1, double6ptr{grad, N});
}

double mpetd_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
        - G (Gravitational constant)
        - lmax
        - num_coeff
        - deg_a (time polynomial degree); len of num_coeff
        - deg_b (time polynomial degree); len of num_coeff
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */
    // double G = pars[0];

    ////until this is fixed, we can just comment all the next lines
//     double G = pars[0];
//     int lmax = (int)pars[1];
//     int num_coeff = (int)pars[2];
//     int deg_a[num_coeff];
//     int deg_b[num_coeff];
//     memset(deg_a, 0, num_coeff * sizeof(int));
//     memset(deg_b, 0, num_coeff * sizeof(int));

//     for(int i=0; i<num_coeff; i++){
//         deg_a[i] = pars[3+i];
//         deg_b[i] = pars[3+num_coeff+i];
//     }

//     double M = pars[2*num_coeff + 3];
//     double r_s = pars[2*num_coeff + 4];


//     // Construct the alm and blm values:
//     double alm[num_coeff], blm[num_coeff];
//     memset(alm, 0, num_coeff * sizeof(double));
//     memset(blm, 0, num_coeff * sizeof(double));


//     double val[1] = {0.};
//     int i, l, m;
//     i = 0;
//     int k_stride = 0;
//     for (l=0; l<(lmax+1); l++) {
//         for (m=0; m<(l+1); m++) {
//             // i = m + (lmax+1) * l;
//             alm[i] = mpetd_polynomial(t, &pars[2*num_coeff + 5 + k_stride],
//                                       deg_a[i]);
//             blm[i] = mpetd_polynomial(t, &pars[2*num_coeff + 5 + k_stride +
//                                                deg_a[i]], deg_b[i]);
//             k_stride += deg_a[i]+deg_b[i];
//             i++;
//         }
//     }

    double val[1] = {0.};

    /* BUG HERE, to do: work out the full density as the laplacian of the
    potential and implement that as a consistency check (should be always 0)
    until then, we set the density to 0
    */
    // mp_density_helper(&q[0], 1,
    //                    M, r_s,
    //                    &alm[0], &blm[0],
    //                    lmax, &val[0]);

    // _val = val[0];
    val[0] = 0.;
    return val[0];
}

/* ---------------------------------------------------------------------------
    Axisymmetric CylSpline (from Agama)

    Parameters
    ----------
    G (Gravitational constant)
    logScaling (whether the interpolated potential is log scaled)
    Rscale (length scale)
    ngridR (number of grid points in R)
    ngridz (number of grid points in z)
    gridR (length `ngridR`) - not actually R values, arcsinh(R/Rscale)
    gridz (length `ngridz`) - not actually z values, arcsinh(z/Rscale)
    gridPhi (length `ngridR` * `ngridz`) - transformed Phi values
    multipole_pars
    -- grid_R, grid_z, grid_Phi to ignore --

    The state is a gsl_spline2d built once by axisym_cylspline_state_alloc().
    Evaluation passes NULL accelerators (binary search) so the shared state is
    never mutated.
*/

static inline double *cylspline_mp_pars(double *pars) {
    int nR = (int)pars[3];
    int nz = (int)pars[4];
    return &pars[5 + nR + nz + nR * nz];
}

void *axisym_cylspline_state_alloc(double *pars) {
    int nR = (int)pars[3];
    int nz = (int)pars[4];
    gsl_spline2d *spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, nR, nz);
    if (spline == NULL)
        return NULL;
    // gsl_spline2d_init copies the grids, so pars need not outlive the spline
    gsl_spline2d_init(spline, &pars[5], &pars[5 + nR], &pars[5 + nR + nz], nR, nz);
    return (void*)spline;
}

void axisym_cylspline_state_free(void *state) {
    if (state != NULL)
        gsl_spline2d_free((gsl_spline2d*)state);
}

/*
    Evaluate the spline at cylindrical (R, z). Fills d with
    {Phi, dPhi/dR, dPhi/dz, d2Phi/dR2, d2Phi/dz2, d2Phi/dRdz}, up to the
    requested derivative order (0, 1, or 2). Returns 0 if (R, z) is outside the
    grid, in which case the caller should use the multipole expansion.
*/
static int cylspline_derivs(void *state, double *pars, double R, double z,
                            int order, double *d) {
    const gsl_spline2d *spline = (const gsl_spline2d*)state;
    int logScaling = (int)pars[1];
    double s = pars[2];
    int nR = (int)pars[3];
    int nz = (int)pars[4];
    double *gridR = &pars[5];
    double *gridz = &pars[5 + nR];

    double u = asinh(R / s);
    double v = asinh(z / s);
    // written so that NaN coordinates also fall through to the multipole
    if (!((u >= gridR[0]) && (u <= gridR[nR-1]) && (v >= gridz[0]) && (v <= gridz[nz-1])))
        return 0;

    double f = gsl_spline2d_eval(spline, u, v, NULL, NULL);
    d[0] = logScaling ? -exp(f) : f;
    if (order < 1)
        return 1;

    // u = asinh(R/s): du/dR = 1/sqrt(s^2+R^2), d2u/dR2 = -R (du/dR)^3
    double uR = 1. / sqrt(s*s + R*R);
    double vz = 1. / sqrt(s*s + z*z);
    double fu = gsl_spline2d_eval_deriv_x(spline, u, v, NULL, NULL);
    double fv = gsl_spline2d_eval_deriv_y(spline, u, v, NULL, NULL);
    double gR = fu * uR;
    double gz = fv * vz;

    double gRR = 0., gzz = 0., gRz = 0.;
    if (order >= 2) {
        gRR = gsl_spline2d_eval_deriv_xx(spline, u, v, NULL, NULL) * uR*uR - fu * R*uR*uR*uR;
        gzz = gsl_spline2d_eval_deriv_yy(spline, u, v, NULL, NULL) * vz*vz - fv * z*vz*vz*vz;
        gRz = gsl_spline2d_eval_deriv_xy(spline, u, v, NULL, NULL) * uR*vz;
    }

    if (logScaling) {
        // Phi = -exp(g): Phi' = Phi g', Phi'' = Phi (g'' + g'^2)
        double Phi = d[0];
        d[1] = Phi * gR;
        d[2] = Phi * gz;
        d[3] = Phi * (gRR + gR*gR);
        d[4] = Phi * (gzz + gz*gz);
        d[5] = Phi * (gRz + gR*gz);
    } else {
        d[1] = gR;
        d[2] = gz;
        d[3] = gRR;
        d[4] = gzz;
        d[5] = gRz;
    }
    return 1;
}

double axisym_cylspline_value(double t, double *pars, double *q, int n_dim, void *state) {
    double d[6];
    double R = sqrt(q[0]*q[0] + q[1]*q[1]);
    if (cylspline_derivs(state, pars, R, q[2], 0, d))
        return d[0];
    return mp_potential(t, cylspline_mp_pars(pars), q, n_dim);
}

void axisym_cylspline_gradient(double t, double *__restrict__ pars, double *__restrict__ q_in, int n_dim, size_t N, double *__restrict__ grad_in, void *__restrict__ state) {
    double6ptr q = double6ptr{q_in, N};
    double6ptr grad = double6ptr{grad_in, N};
    double d[6];

    for(size_t i = 0; i < N; i++) {
        double R = sqrt(q.x[i]*q.x[i] + q.y[i]*q.y[i]);

        if (cylspline_derivs(state, pars, R, q.z[i], 1, d)) {
            if (R > 0) {
                grad.x[i] += d[1] * q.x[i] / R;
                grad.y[i] += d[1] * q.y[i] / R;
            }
            grad.z[i] += d[2];
        } else {  // Use external Multipole
            _mp_gradient_single(t, cylspline_mp_pars(pars), double6ptr{q_in + i, N}, n_dim, double6ptr{grad_in + i, N}, NULL);
        }
    }
}

void axisym_cylspline_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    double d[6];
    double x = q[0], y = q[1];
    double R = sqrt(x*x + y*y);

    if (cylspline_derivs(state, pars, R, q[2], 2, d)) {
        double PR = d[1], PRR = d[3], Pzz = d[4], PRz = d[5];
        double Hxx, Hyy, Hxy, Hxz, Hyz;
        if (R > 0) {
            double R2 = R*R;
            Hxx = PRR * x*x / R2 + PR * y*y / (R2*R);
            Hyy = PRR * y*y / R2 + PR * x*x / (R2*R);
            Hxy = (PRR - PR / R) * x*y / R2;
            Hxz = PRz * x / R;
            Hyz = PRz * y / R;
        } else {
            Hxx = Hyy = PRR;
            Hxy = Hxz = Hyz = 0.;
        }
        hess[0] += Hxx; hess[1] += Hxy; hess[2] += Hxz;
        hess[3] += Hxy; hess[4] += Hyy; hess[5] += Hyz;
        hess[6] += Hxz; hess[7] += Hyz; hess[8] += Pzz;

    } else {
        // ponytail: central finite difference of the multipole gradient outside
        // the grid; add an analytic mp hessian if accuracy there matters
        double r = sqrt(R*R + q[2]*q[2]);
        double h = 1e-5 * r;
        for (int j = 0; j < 3; j++) {
            double qp[6] = {q[0], q[1], q[2], 0, 0, 0};
            double qm[6] = {q[0], q[1], q[2], 0, 0, 0};
            double gp[6] = {0.}, gm[6] = {0.};
            qp[j] += h;
            qm[j] -= h;
            _mp_gradient_single(t, cylspline_mp_pars(pars), double6ptr{qp, 1}, n_dim, double6ptr{gp, 1}, NULL);
            _mp_gradient_single(t, cylspline_mp_pars(pars), double6ptr{qm, 1}, n_dim, double6ptr{gm, 1}, NULL);
            for (int i = 0; i < 3; i++)
                hess[3*i + j] += (gp[i] - gm[i]) / (2*h);
        }
    }
}

double axisym_cylspline_density(double t, double *pars, double *q, int n_dim, void *state) {
    double G = pars[0];
    double d[6];
    double R = sqrt(q[0]*q[0] + q[1]*q[1]);

    if (cylspline_derivs(state, pars, R, q[2], 2, d)) {
        // Laplacian in cylindrical coords; dPhi/dR / R -> d2Phi/dR2 as R -> 0
        double PR_R = (R > 0) ? d[1] / R : d[3];
        return (PR_R + d[3] + d[4]) / (4 * M_PI * G);
    }
    return mp_density(t, cylspline_mp_pars(pars), q, n_dim);
}

#endif  // USE_GSL
