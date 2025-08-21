# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import numpy as np
cimport numpy as np
np.import_array()

from libcpp.string cimport string
from libcpp cimport bool as cbool

from ..cpotential cimport CPotentialWrapper
from ..cpotential cimport densityfunc, energyfunc, gradientfunc, hessianfunc

cdef extern from "extra_compile_macros.h":
    int USE_EXP

cdef extern from "potential/potential/builtin/exp_fields.h" namespace "gala_exp":
    cdef cppclass State:
        double tmin
        double tmax
        cbool is_static

    State exp_init(
        const string &config,
        const string &coeffile,
        int stride,
        double tmin,
        double tmax,
        int snapshot_index
    ) except + nogil

cdef extern from "potential/potential/builtin/exp_fields.h":
    # TODO: technically these are C++ signatures, unlike the extern "C" declarations in exp_fields.h
    double exp_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double exp_density(double t, double *pars, double *q, int n_dim, void *state) nogil

__all__ = [
    'EXPWrapper',
]

# ==============================================================================
# EXP potential
#

cdef class EXPWrapper(CPotentialWrapper):
    cdef State exp_state

    def __init__(self, G, parameters, q0, R, config_file, coef_file, stride, snapshot_index):
        tmin = parameters[0]
        tmax = parameters[1]

        self.init(
            [G],
            np.ascontiguousarray(q0),
            np.ascontiguousarray(R)
        )

        if USE_EXP == 1:
            self.exp_state = exp_init(
                str(config_file),
                str(coef_file),
                stride,
                tmin,
                tmax,
                snapshot_index
            )
            self.cpotential.state[0] = &self.exp_state
            self.cpotential.value[0] = <energyfunc>(exp_value)
            self.cpotential.density[0] = <densityfunc>(exp_density)
            self.cpotential.gradient[0] = <gradientfunc>(exp_gradient)


    @property
    def static(self):
        return self.exp_state.is_static

    @property
    def tmin(self):
        return self.exp_state.tmin

    @property
    def tmax(self):
        return self.exp_state.tmax
