/*
    Minimal hand-off struct for delivering a stochastic "kick" into an integrator.

    This header is intentionally tiny and GSL-free: the leapfrog integrator only needs
    this struct layout so it can call the kick through a function pointer (a cross-
    extension call), without compiling or linking any of the diffusion / GSL machinery.
    The diffusion extension fills in the function pointer, state, and RNG.
*/
#ifndef _GALA_DIFFUSION_KICK_H
#define _GALA_DIFFUSION_KICK_H

/*
    Compute the phase-space increment (dx, dv) for a single orbit at the synchronized
    state (t, x, v). x, v, dx, dv are length-3 (Cartesian). dt is the step. `state` and
    `rng` are opaque to the caller.
*/
typedef void (*kickfunc)(double t, double dt, double *x, double *v,
                         double *dx, double *dv, void *state, void *rng);

typedef struct {
    kickfunc kick;  // the kick implementation (lives in the diffusion extension)
    void *state;    // model state (a CDiffusion*)
    void *rng;      // random-number generator (a gsl_rng*)
} DiffusionKick;

#endif
