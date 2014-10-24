cdef class _CPotential:
    cpdef value(self, double[:,::1] q, double[::1] pot)
    cdef public double _value(self, double[:,::1] q, int k) nogil

    cpdef gradient(self, double[:,::1] q)
    cdef public void _gradient(self, double[:,::1] q, double[:,::1] grad, int k) nogil

    cpdef hessian(self, double[:,::1] w)
    cdef public void _hessian(self, double[:,::1] w, double[:,:,::1] hess, int k) nogil

    cpdef mass_enclosed(self, double[:,::1] q)
    cdef public double _mass_enclosed(self, double[:,::1] q, double[:,::1] epsilon, double Gee, int k)
