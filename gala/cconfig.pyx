# cython: language_level=3

cdef extern from "extra_compile_macros.h":
    int USE_GSL

if USE_GSL == 1:
    GSL_ENABLED = True
else:
    GSL_ENABLED = False
