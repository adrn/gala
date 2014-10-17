cdef class _CPotential:
    cpdef value(self, double[:,::1] q)
    cdef public void _value(self, double[:,::1] q, double[::1] pot, int nparticles) nogil

    cpdef gradient(self, double[:,::1] q)
    cdef public void _gradient(self, double[:,::1] q, double[:,::1] grad, int nparticles) nogil

    cpdef hessian(self, double[:,::1] w)
    cdef public void _hessian(self, double[:,::1] w, double[:,::1] hess, int nparticles) nogil

    cpdef mass_enclosed(self, double[:,::1] q)
    cdef public void _mass_enclosed(self, double[:,::1] q, double[::1] mass, int nparticles)
