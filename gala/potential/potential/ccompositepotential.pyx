# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3


import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

from libc.stdio cimport printf


from .core import CompositePotential
from .cpotential import CPotentialBase
from .cpotential cimport (
    CPotentialWrapper, CPotential, allocate_cpotential, free_cpotential, resize_cpotential_arrays
)

__all__ = ['CCompositePotential']

cdef class CCompositePotentialWrapper(CPotentialWrapper):
    def __init__(self, list potentials):
        cdef:
            int i, n_components
            CPotentialWrapper[::1] _cpotential_arr

        self._potentials = potentials
        _cpotential_arr = np.array(potentials)
        n_components = len(potentials)

        # First, check if we need more components
        if n_components > 1:
            # Reallocate arrays without freeing the struct itself
            # (This requires implementing a resize function in the C code)
            resize_cpotential_arrays(self.cpotential, n_components)

        # Store parameter counts
        self._n_params = np.zeros(n_components, dtype=np.int32)
        for i in range(n_components):
            self._n_params[i] = _cpotential_arr[i]._n_params[0]
            self.cpotential.n_params[i] = self._n_params[i]
            self.cpotential.do_shift_rotate[i] = _cpotential_arr[i].cpotential.do_shift_rotate[0]

        self.cpotential.n_dim = 0

        for i in range(n_components):
            self.cpotential.parameters[i] = &(_cpotential_arr[i]._params[0])
            self.cpotential.q0[i] = &(_cpotential_arr[i]._q0[0])
            self.cpotential.R[i] = &(_cpotential_arr[i]._R[0])
            self.cpotential.state[i] = _cpotential_arr[i].cpotential.state[0]
            self.cpotential.value[i] = _cpotential_arr[i].cpotential.value[0]
            self.cpotential.density[i] = _cpotential_arr[i].cpotential.density[0]
            self.cpotential.gradient[i] = _cpotential_arr[i].cpotential.gradient[0]
            self.cpotential.hessian[i] = _cpotential_arr[i].cpotential.hessian[0]

            if self.cpotential.n_dim == 0:
                self.cpotential.n_dim = _cpotential_arr[i].cpotential.n_dim
            elif self.cpotential.n_dim != _cpotential_arr[i].cpotential.n_dim:
                raise ValueError(
                    "Input potentials must have same number of coordinate dimensions"
                )

    def __reduce__(self):
        return (self.__class__, (list(self._potentials),))


class CCompositePotential(CompositePotential, CPotentialBase):

    def __init__(self, **potentials):
        CompositePotential.__init__(self, **potentials)

    def _reset_c_instance(self):
        """Rebuilds the C instance after the composite potential is modified."""
        self._potential_list = []
        for p in self.values():
            self._potential_list.append(p.c_instance)

        if len(self._potential_list) > 0:
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
