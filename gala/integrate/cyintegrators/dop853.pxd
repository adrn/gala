cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef void dop853_step(CPotential *cp, CFrame *cf,
                      double *w, double t1, double t2, double dt0,
                      int ndim, int norbits,
                      double atol, double rtol, int nmax)

cdef dop853_helper(CPotential *cp, CFrame *cf,
                   double[:,::1] w0, double[::1] t,
                   int ndim, int norbits, int ntimes,
                   double atol, double rtol, int nmax)

cdef dop853_helper_save_all(CPotential *cp, CFrame *cf,
                            double[:,::1] w0, double[::1] t,
                            int ndim, int norbits, int ntimes,
                            double atol, double rtol, int nmax)

# cpdef dop853_integrate_hamiltonian(hamiltonian, double[:,::1] w0, double[::1] t,
#                                    double atol=?, double rtol=?, int nmax=?)
