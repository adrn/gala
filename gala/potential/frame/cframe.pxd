cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef class CFrameWrapper:
    cdef CFrame cframe

