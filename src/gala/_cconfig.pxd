# cython: language_level=3
# cython: language=c++

cdef extern from "extra_compile_macros.h":
    int USE_GSL
    int USE_EXP
