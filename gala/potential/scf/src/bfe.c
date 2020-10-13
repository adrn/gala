#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bfe_helper.h"
#include "extra_compile_macros.h"

#if USE_GSL == 1
void scf_density_helper(double *xyz, int K,
                        double M, double r_s,
                        double *Snlm, double *Tnlm,
                        int nmax, int lmax, double *dens) {

    int i,j,k, n,l,m;
    double r, s, X, phi;
    double cosmphi[lmax+1], sinmphi[lmax+1];
    memset(cosmphi, 0, (lmax+1)*sizeof(double));
    memset(sinmphi, 0, (lmax+1)*sizeof(double));
    for (k=0; k<K; k++) {
        j = 3*k;
        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        s = r/r_s;
        X = xyz[j+2]/r; // cos(theta)
        phi = atan2(xyz[j+1], xyz[j+0]);

        // precompute all cos(m phi), sin(m phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        // i = 0;
        for (n=0; n<(nmax+1); n++) {
            for (l=0; l<(lmax+1); l++) {
                for (m=0; m<(lmax+1); m++) {
                    if (m > l) {
                        // i++;
                        continue;
                    }

                    i = m + (lmax+1) * (l + (lmax+1) * n);
                    if ((Snlm[i] == 0.) & (Tnlm[i] == 0.)) {
                        // i++;
                        continue;
                    }
                    dens[k] += rho_nlm(s, phi, X, n, l, m) * (Snlm[i]*cosmphi[m] +
                                                              Tnlm[i]*sinmphi[m]);
                }
            }
        }
        dens[k] *= M / (r_s*r_s*r_s);
    }
}

void scf_potential_helper(double *xyz, int K,
                          double G, double M, double r_s,
                          double *Snlm, double *Tnlm,
                          int nmax, int lmax, double *val) {

    int i,j,k, n,l,m;
    double r, s, X, phi;
    double cosmphi[lmax+1], sinmphi[lmax+1];
    memset(cosmphi, 0, (lmax+1)*sizeof(double));
    memset(sinmphi, 0, (lmax+1)*sizeof(double));
    for (k=0; k<K; k++) {
        j = 3*k;
        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        s = r/r_s;
        X = xyz[j+2]/r; // cos(theta)
        phi = atan2(xyz[j+1], xyz[j+0]);

        // HACK: zero out before filling;
        val[k] = 0.;

        // precompute all cos(m phi), sin(m phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        // TODO: could speed this up by moving call to legendre out of n loop
        // TODO: note, if I do this I need to go from 3D to 1D array in different way...
        // i = 0;
        for (n=0; n<(nmax+1); n++) {
            for (l=0; l<(lmax+1); l++) {
                for (m=0; m<(lmax+1); m++) {
                    if (m > l) {
                        // i++;
                        continue;
                    }

                    i = m + (lmax+1) * (l + (lmax+1) * n);
                    if ((Snlm[i] == 0.) & (Tnlm[i] == 0.)) {
                        // i++;
                        continue;
                    }

                    val[k] += phi_nlm(s, phi, X, n, l, m) * (Snlm[i]*cosmphi[m] +
                                                             Tnlm[i]*sinmphi[m]);
                    // i++;
                }
            }
        }
        val[k] *= G*M/r_s;
    }
}

void scf_gradient_helper(double *xyz, int K,
                         double G, double M, double r_s,
                         double *Snlm, double *Tnlm,
                         int nmax, int lmax, double *grad) {

    int i,j,k, n,l,m;
    double r, s, X, phi;
    double sintheta, cosphi, sinphi, tmp;
    double tmp_grad[3], tmp_grad2[3*K]; // TODO: this might be really inefficient
    double cosmphi[lmax+1], sinmphi[lmax+1];
    memset(cosmphi, 0, (lmax+1)*sizeof(double));
    memset(sinmphi, 0, (lmax+1)*sizeof(double));

    for (k=0; k<K; k++) {
        j = 3*k;
        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        s = r/r_s;
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

        // i = 0;
        for (n=0; n<(nmax+1); n++) {
            // gsl_sf_legendre_deriv_array(GSL_SF_LEGENDRE_SPHARM, lmax, X,
            //                             double result_array[], double result_deriv_array[]);
            for (l=0; l<(lmax+1); l++) {
                for (m=0; m<(lmax+1); m++) {
                    if (m > l) {
                        // i++;
                        continue;
                    }

                    i = m + (lmax+1) * (l + (lmax+1) * n);
                    tmp = (Snlm[i]*cosmphi[m] + Tnlm[i]*sinmphi[m]);
                    if ((Snlm[i] == 0.) & (Tnlm[i] == 0.)) {
                        // i++;
                        continue;
                    }

                    sph_grad_phi_nlm(s, phi, X, n, l, m, lmax, &tmp_grad[0]);
                    tmp_grad2[j+0] += tmp_grad[0] * tmp; // r
                    tmp_grad2[j+1] += tmp_grad[1] * tmp; // theta
                    tmp_grad2[j+2] += tmp_grad[2] * (Tnlm[i]*cosmphi[m] - Snlm[i]*sinmphi[m]) / (s*sintheta); // phi

                    // i++;
                }
            }
        }
        tmp_grad[0] = tmp_grad2[j+0];
        tmp_grad[1] = tmp_grad2[j+1];
        tmp_grad[2] = tmp_grad2[j+2];

        // transform to cartesian
        tmp_grad2[j+0] = sintheta*cosphi*tmp_grad[0] + X*cosphi*tmp_grad[1] - sinphi*tmp_grad[2];
        tmp_grad2[j+1] = sintheta*sinphi*tmp_grad[0] + X*sinphi*tmp_grad[1] + cosphi*tmp_grad[2];
        tmp_grad2[j+2] = X*tmp_grad[0] - sintheta*tmp_grad[1];

        grad[j+0] = grad[j+0] + tmp_grad2[j+0]*G*M/(r_s*r_s);
        grad[j+1] = grad[j+1] + tmp_grad2[j+1]*G*M/(r_s*r_s);
        grad[j+2] = grad[j+2] + tmp_grad2[j+2]*G*M/(r_s*r_s);
    }
}

double scf_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
        - G (Gravitational constant)
        - nmax
        - lmax
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */

    double G = pars[0];
    int nmax = (int)pars[1];
    int lmax = (int)pars[2];
    double M = pars[3];
    double r_s = pars[4];

    double val[1] = {0.};
    double _val;
    int n,l,m;

    int num_coeff = 0;
    for (n=0; n<(nmax+1); n++) {
        for (l=0; l<(lmax+1); l++) {
            for (m=0; m<(lmax+1); m++) {
                num_coeff++;
            }
        }
    }

    scf_potential_helper(&q[0], 1,
                         G, M, r_s,
                         &pars[5], &pars[5+num_coeff],
                         nmax, lmax, &val[0]);

    _val = val[0];
    return _val;
}

void scf_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
        - G (Gravitational constant)
        - nmax
        - lmax
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */
    double G = pars[0];
    int nmax = (int)pars[1];
    int lmax = (int)pars[2];
    double M = pars[3];
    double r_s = pars[4];

    int n,l,m;

    int num_coeff = 0;
    for (n=0; n<(nmax+1); n++) {
        for (l=0; l<(lmax+1); l++) {
            for (m=0; m<(lmax+1); m++) {
                num_coeff++;
            }
        }
    }

    scf_gradient_helper(&q[0], 1,
                        G, M, r_s,
                        &pars[5], &pars[5+num_coeff],
                        nmax, lmax, &grad[0]);
}

double scf_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
        - G (Gravitational constant)
        - nmax
        - lmax
        - m (mass scale)
        - r_s (length scale)
        [- sin_coeff, cos_coeff]
    */
    double G = pars[0];
    int nmax = (int)pars[1];
    int lmax = (int)pars[2];
    double M = pars[3];
    double r_s = pars[4];

    double val[1] = {0.};
    double _val;
    int n,l,m;

    int num_coeff = 0;
    for (n=0; n<(nmax+1); n++) {
        for (l=0; l<(lmax+1); l++) {
            for (m=0; m<(lmax+1); m++) {
                num_coeff++;
            }
        }
    }

    scf_density_helper(&q[0], 1,
                       M, r_s,
                       &pars[5], &pars[5+num_coeff],
                       nmax, lmax, &val[0]);

    _val = val[0];
    return _val;
}
#endif
