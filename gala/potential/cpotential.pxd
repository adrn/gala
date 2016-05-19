cdef extern from "src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef class CPotentialWrapper:
    cdef CPotential cpotential
    cdef double[::1] _params
    cdef int[::1] _n_params

    cpdef value(self, double[:,::1] q, double t=?)
    cpdef density(self, double[:,::1] q, double t=?)
    cpdef gradient(self, double[:,::1] q, double t=?)
    # cpdef hessian(self, double[:,::1] q, double t=?)

    cpdef d_dr(self, double[:,::1] q, double G, double t=?)
    cpdef d2_dr2(self, double[:,::1] q, double G, double t=?)
    cpdef mass_enclosed(self, double[:,::1] q, double G, double t=?)
