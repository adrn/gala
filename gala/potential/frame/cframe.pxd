cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef class CFrameWrapper:
    cdef CFrame cframe
    cdef double[::1] _params
    cpdef energy(self, double[:,::1] w, double[::1] t)
    cpdef gradient(self, double[:,::1] w, double[::1] t)
    cpdef hessian(self, double[:,::1] w, double[::1] t)

