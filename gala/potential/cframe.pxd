cdef extern from "src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef class CFrameWrapper:
    cdef CFrame cframe

