# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

from libc.stdio cimport printf

# Project
from .core import CompositePotential
from .cpotential import CPotentialBase
from .cpotential cimport CPotentialWrapper, CPotential

__all__ = ['CCompositePotential']

cdef class CCompositePotentialWrapper(CPotentialWrapper):

    def __init__(self, list potentials):
        cdef:
            CPotential cp
            CPotential tmp_cp
            int i
            CPotentialWrapper[::1] _cpotential_arr

        self._potentials = potentials
        _cpotential_arr = np.array(potentials)

        n_components = len(potentials)
        self._n_params = np.zeros(n_components, dtype=np.int32)
        for i in range(n_components):
            self._n_params[i] = _cpotential_arr[i]._n_params[0]

        cp.n_components = n_components
        cp.n_params = &(self._n_params[0])
        cp.n_dim = 0

        for i in range(n_components):
            tmp_cp = _cpotential_arr[i].cpotential
            cp.parameters[i] = &(_cpotential_arr[i]._params[0])
            cp.q0[i] = &(_cpotential_arr[i]._q0[0])
            cp.R[i] = &(_cpotential_arr[i]._R[0])
            cp.value[i] = tmp_cp.value[0]
            cp.density[i] = tmp_cp.density[0]
            cp.gradient[i] = tmp_cp.gradient[0]
            cp.hessian[i] = tmp_cp.hessian[0]

            if cp.n_dim == 0:
                cp.n_dim = tmp_cp.n_dim
            elif cp.n_dim != tmp_cp.n_dim:
                raise ValueError("Input potentials must have same number of coordinate dimensions")

        self.cpotential = cp

    def __reduce__(self):
        return (self.__class__, (list(self._potentials),))


class CCompositePotential(CompositePotential, CPotentialBase):

    def __init__(self, **potentials):
        CompositePotential.__init__(self, **potentials)

    def _reset_c_instance(self):
        self._potential_list = []
        for p in self.values():
            self._potential_list.append(p.c_instance)
        self.G = p.G
        self.c_instance = CCompositePotentialWrapper(self._potential_list)

    def __setitem__(self, *args, **kwargs):
        CompositePotential.__setitem__(self, *args, **kwargs)
        self._reset_c_instance()

    def __setstate__(self, state):
        # when rebuilding from a pickle, temporarily release lock
        self.lock = False
        self._units = None
        for name, potential in state:
            self[name] = potential
        self._reset_c_instance()
        self.lock = True

    def __reduce__(self):
        """ Properly package the object for pickling """
        return self.__class__, (), list(self.items())
