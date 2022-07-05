/*
copied from https://github.com/adrn/gala/blob/main/gala/potential/scf/src/bfe.c
and https://github.com/adrn/gala/blob/main/gala/potential/scf/src/bfe_helper.c
*/
#include <stdlib.h>
#include <stdio.h>
#include "extra_compile_macros.h"
#include <math.h>
#include <string.h>

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
                        int lmax, int inner, double *sphgrad) {
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
        sphgrad[0] = dPhil_dr;
        sphgrad[1] = dPhi_dtheta;
        sphgrad[2] = dPhi_dphi;
    } else {
        sphgrad[0] = 0;
        sphgrad[1] = 0;
        sphgrad[2] = 0;
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

void mp_gradient_helper(double *xyz, int K,
                        double G, double M, double r_s,
                        double *Slm, double *Tlm,
                        int lmax, int inner, double *grad) {

    int i,j,k, l,m;
    double s, r, X, phi;
    double sintheta, cosphi, sinphi, tmp;
    double tmp_grad[3], tmp_grad2[3*K]; // TODO: this might be really inefficient
    double cosmphi[lmax+1], sinmphi[lmax+1];

    for (k=0; k<K; k++) {
        j = 3*k;
        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        s = r / r_s;
        X = xyz[j+2]/r; // cos(theta)
        phi = atan2(xyz[j+1], xyz[j+0]);

        sintheta = sqrt(1 - X*X);
        cosphi = cos(phi);
        sinphi = sin(phi);

        // precompute all cos(m phi), sin(m phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        // zero out
        tmp_grad2[j+0] = 0.;
        tmp_grad2[j+1] = 0.;
        tmp_grad2[j+2] = 0.;

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

                mp_sph_grad_phi_lm(s, phi, X, l, m, lmax, inner, &tmp_grad[0]);
                tmp_grad2[j+0] += tmp_grad[0] * tmp; // r
                tmp_grad2[j+1] += tmp_grad[1] * tmp; // phi??

                if (sintheta != 0) {
                    tmp_grad2[j+2] += tmp_grad[2] * (
                        Tlm[i]*cosmphi[m] - Slm[i]*sinmphi[m]
                    ) / (s * sintheta); // theta??
                } else {
                    tmp_grad2[j+2] = 0.;
                }


                i++;
            }
        }
        tmp_grad[0] = tmp_grad2[j+0];
        tmp_grad[1] = tmp_grad2[j+1];
        tmp_grad[2] = tmp_grad2[j+2];

        // transform to cartesian
        tmp_grad2[j+0] = sintheta*cosphi*tmp_grad[0] + X*cosphi*tmp_grad[1]
            - sinphi*tmp_grad[2];
        tmp_grad2[j+1] = sintheta*sinphi*tmp_grad[0] + X*sinphi*tmp_grad[1]
            + cosphi*tmp_grad[2];
        tmp_grad2[j+2] = X*tmp_grad[0] - sintheta*tmp_grad[1];

        grad[j+0] = grad[j+0] + tmp_grad2[j+0] * G*M/(r_s*r_s);
        grad[j+1] = grad[j+1] + tmp_grad2[j+1] * G*M/(r_s*r_s);
        grad[j+2] = grad[j+2] + tmp_grad2[j+2] * G*M/(r_s*r_s);
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

void mp_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
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

    mp_gradient_helper(&q[0], 1,
                       G, M, r_s,
                       &Slm[0], &Tlm[0],
                       lmax, inner, &grad[0]);
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

void mpetd_gradient(double t, double *pars, double *q,
                    int n_dim, double *grad) {
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



    mp_gradient_helper(&q[0], 1,
                        G, M, r_s,
                        &alm[0], &blm[0],
                        lmax, 1, &grad[0]);
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
*/

double axisym_cylspline_value(double t, double *pars, double *q, int n_dim) {
    int logScaling = (int)pars[1];
    double Rscale = pars[2];
    int nR = (int)pars[3];
    int nz = (int)pars[4];

    double Phi;
    double Rasinh = sqrt(q[0]*q[0] + q[1]*q[1]);
    double zasinh = q[2];
    Rasinh = asinh(Rasinh / Rscale);
    zasinh = asinh(zasinh / Rscale);

    double gridR[nR];
    double gridz[nz];
    double gridPhi[nz * nR];
    for (int i=0; i < nR; i++)
        gridR[i] = pars[5 + i];
    for (int i=0; i < nz; i++)
        gridz[i] = pars[5 + nR + i];
    for (int i=0; i < nR; i++)
        for (int j=0; j < nz; j++)
            gridPhi[i * nz + j] = pars[5 + nR + nz + i * nz + j];

    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nR, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *yacc = gsl_interp_accel_alloc();

    // TODO: interpolation is very slow I think because this setup is done every
    // time the function is called...

    if ((Rasinh >= gridR[0]) && (Rasinh <= gridR[nR-1]) &&
        (zasinh >= gridz[0]) && (zasinh <= gridz[nz-1])) { // Use CylSpline

        /* initialize interpolation */
        // TODO: define this in wrapper, make all CPotential's have a void
        // pointer array to store things like this, all these functions then
        // need to accept one more parameter (or is there a way to do optional
        // args in C?), ??, profit.
        gsl_spline2d_init(spline, gridR, gridz, gridPhi, nR, nz);
        Phi = gsl_spline2d_eval(spline, Rasinh, zasinh, xacc, yacc);

        if (logScaling)
            Phi = -exp(Phi);

    } else {  // Use external Multipole
        Phi = mp_potential(t, &pars[5 + nR + nz + nR * nz], q, n_dim);
    }

    gsl_spline2d_free(spline);
    gsl_interp_accel_free(xacc);
    gsl_interp_accel_free(yacc);

    return Phi;
}

void axisym_cylspline_gradient(double t, double *pars, double *q, int n_dim,
                               double *grad) {

    int logScaling = (int)pars[1];
    double Rscale = pars[2];
    int nR = (int)pars[3];
    int nz = (int)pars[4];

    double Phi, dPhi_dR, dPhi_dz;
    double R = sqrt(q[0]*q[0] + q[1]*q[1]);
    double Rasinh = asinh(R / Rscale);
    double zasinh = asinh(q[2] / Rscale);

    double gridR[nR];
    double gridz[nz];
    double gridPhi[nz * nR];
    for (int i=0; i < nR; i++)
        gridR[i] = pars[5 + i];
    for (int i=0; i < nz; i++)
        gridz[i] = pars[5 + nR + i];
    for (int i=0; i < nR; i++)
        for (int j=0; j < nz; j++)
            gridPhi[i * nz + j] = pars[5 + nR + nz + i * nz + j];

    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nR, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *yacc = gsl_interp_accel_alloc();

    // TODO: interpolation is very slow I think because this setup is done every
    // time the function is called...

    if ((Rasinh >= gridR[0]) && (Rasinh <= gridR[nR-1]) &&
        (zasinh >= gridz[0]) && (zasinh <= gridz[nz-1])) { // Use CylSpline

        /* initialize interpolation */
        // TODO: define this in wrapper, make all CPotential's have a void
        // pointer array to store things like this, all these functions then
        // need to accept one more parameter (or is there a way to do optional
        // args in C?), ??, profit.
        gsl_spline2d_init(spline, gridR, gridz, gridPhi, nR, nz);

        dPhi_dR = gsl_spline2d_eval_deriv_x(spline, Rasinh, zasinh, xacc, yacc);
        dPhi_dR = dPhi_dR / (Rscale * cosh(Rasinh));

        dPhi_dz = gsl_spline2d_eval_deriv_y(spline, Rasinh, zasinh, xacc, yacc);
        dPhi_dz = dPhi_dz / (Rscale * cosh(zasinh));

        if (logScaling) {
            Phi = gsl_spline2d_eval(spline, Rasinh, zasinh, xacc, yacc);
            Phi = -exp(Phi);
            dPhi_dR = dPhi_dR * Phi;
            dPhi_dz = dPhi_dz * Phi;
        }

        if (R > 0) {
            grad[0] = grad[0] + dPhi_dR * q[0] / R;
            grad[1] = grad[1] + dPhi_dR * q[1] / R;
            grad[2] = grad[2] + dPhi_dz;
        } else {
            grad[2] = grad[2] + dPhi_dz;
        }

    } else {  // Use external Multipole
        mp_gradient(t, &pars[5 + nR + nz + nR * nz], q, n_dim, grad);
    }
    gsl_spline2d_free(spline);
    gsl_interp_accel_free(xacc);
    gsl_interp_accel_free(yacc);
}

double axisym_cylspline_density(double t, double *pars, double *q, int n_dim) {
    double G = pars[0];
    int logScaling = (int)pars[1];
    double Rscale = pars[2];
    int nR = (int)pars[3];
    int nz = (int)pars[4];

    return 0.0/0.0;

    /* TODO: bug in the below... */

    double dens;
    double Phi, dPhi_dR, dPhi_dz, d2Phi_dR2, d2Phi_dz2;
    double R = sqrt(q[0]*q[0] + q[1]*q[1]);
    double Rasinh = asinh(R / Rscale);
    double zasinh = asinh(q[2] / Rscale);

    double gridR[nR];
    double gridz[nz];
    double gridPhi[nz * nR];
    for (int i=0; i < nR; i++)
        gridR[i] = pars[5 + i];
    for (int i=0; i < nz; i++)
        gridz[i] = pars[5 + nR + i];
    for (int i=0; i < nR; i++)
        for (int j=0; j < nz; j++)
            gridPhi[i * nz + j] = pars[5 + nR + nz + i * nz + j];

    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nR, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *yacc = gsl_interp_accel_alloc();

    // TODO: interpolation is very slow I think because this setup is done every
    // time the function is called...

    if ((Rasinh >= gridR[0]) && (Rasinh <= gridR[nR-1]) &&
        (zasinh >= gridz[0]) && (zasinh <= gridz[nz-1])) { // Use CylSpline

        /* initialize interpolation */
        // TODO: define this in wrapper, make all CPotential's have a void
        // pointer array to store things like this, all these functions then
        // need to accept one more parameter (or is there a way to do optional
        // args in C?), ??, profit.
        gsl_spline2d_init(spline, gridR, gridz, gridPhi, nR, nz);

        dPhi_dR = gsl_spline2d_eval_deriv_x(spline, Rasinh, zasinh, xacc, yacc);
        dPhi_dR = dPhi_dR / (Rscale * cosh(Rasinh));

        dPhi_dz = gsl_spline2d_eval_deriv_y(spline, Rasinh, zasinh, xacc, yacc);
        dPhi_dz = dPhi_dz / (Rscale * cosh(zasinh));

        d2Phi_dR2 = gsl_spline2d_eval_deriv_xx(spline, Rasinh, zasinh, xacc, yacc);
        d2Phi_dR2 = d2Phi_dR2 / pow(Rscale * cosh(Rasinh), 2);

        d2Phi_dz2 = gsl_spline2d_eval_deriv_yy(spline, Rasinh, zasinh, xacc, yacc);
        d2Phi_dz2 = d2Phi_dz2 / pow(Rscale * cosh(zasinh), 2);

        if (logScaling) {
            Phi = gsl_spline2d_eval(spline, Rasinh, zasinh, xacc, yacc);
            Phi = -exp(Phi);
            dPhi_dR = dPhi_dR * Phi;
            d2Phi_dR2 = (d2Phi_dR2 + pow(dPhi_dR / Phi, 2)) * Phi;

            dPhi_dz = dPhi_dz * Phi;
            d2Phi_dz2 = (d2Phi_dz2 + pow(dPhi_dz / Phi, 2)) * Phi;
        }

        dens = (dPhi_dR / R + d2Phi_dR2 + d2Phi_dz2) / (4 * M_PI * G);

    } else {  // Use external Multipole
        dens = mp_density(t, &pars[5 + nR + nz + nR * nz], q, n_dim);
    }
    gsl_spline2d_free(spline);
    gsl_interp_accel_free(xacc);
    gsl_interp_accel_free(yacc);

    return dens;
}



#endif