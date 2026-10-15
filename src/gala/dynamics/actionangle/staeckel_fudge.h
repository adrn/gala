#ifndef _STAECKEL_FUDGE_H
#define _STAECKEL_FUDGE_H

#include <stddef.h>
#include "potential/src/cpotential.h"

/*
    Actions, angles, and frequencies from the Staeckel fudge (Binney 2012, Sanders
    2012). This somewhat follows the implementation in Agama (Vasiliev 2019), but I made
    some small changes. For example, I changed the "change of variables" used to compute
    integrals stably to use sin(u) instead of a cubic polynomial.

    w: (n, 6) Cartesian phase-space positions (C-contiguous)
    delta: (n,) focal distances
    out: (n, 9) = [Jr, Jphi, Jz, theta_r, theta_phi, theta_z, Omega_r, Omega_phi, Omega_z]

    Note that this requires GSL.
*/
void staeckel_fudge(
    CPotential *p, double t, const double *w, const double *delta, size_t n, double *out
);

#endif
