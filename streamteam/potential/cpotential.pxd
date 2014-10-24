cdef class _CPotential:
    cpdef value(self, double[:,::1] q, double[::1] pot)
    cdef double _value(self, double[:,::1] q, int k)

    cpdef gradient(self, double[:,::1] q)
    cdef void _gradient(self, double[:,::1] q, double[:,::1] grad, int k) nogil

    cpdef hessian(self, double[:,::1] w)
    cdef void _hessian(self, double[:,::1] w, double[:,:,::1] hess, int k) nogil

    cpdef mass_enclosed(self, double[:,::1] q)
    cdef double _mass_enclosed(self, double[:,::1] q, double[:,::1] epsilon, double Gee, int k)
