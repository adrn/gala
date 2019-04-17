# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

# Standard library
import warnings

# Third-party
from astropy.constants import G
import astropy.units as u

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt
from cpython.exc cimport PyErr_CheckSignals

from ...potential import Hamiltonian, NullPotential, CCompositePotential
from ...units import galactic, UnitSystem
from ...integrate.timespec import parse_time_specification
from ..orbit import Orbit

from ...potential.potential.cpotential cimport CPotentialWrapper
from ...potential.frame.cframe cimport CFrameWrapper
from ...integrate.cyintegrators.dop853 cimport dop853_helper_save_all

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef extern from "potential/src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef struct CPotential:
        int n_components
        int n_dim
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrame *fr, unsigned norbits,
                              void *args) nogil
    void Fwrapper_direct_nbody(unsigned ndim, double t, double *w, double *f,
                               CPotential *p, CFrame *fr, unsigned norbits,
                               void *args)

DEF MAX_NBODY = 1024;

cpdef _direct_nbody_dop853(double [:, ::1] w0, double[::1] t,
                           hamiltonian, list particle_potentials,
                           double atol=1E-10, double rtol=1E-10, int nmax=0):
    """
    TODO
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("Input must be a Hamiltonian object, not {}"
                        .format(type(hamiltonian)))

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    cdef:
        unsigned nparticles = w0.shape[0]
        unsigned ndim = w0.shape[1]
        unsigned ntimes = len(t)

        int i
        void *args
        CPotential *c_particle_potentials[MAX_NBODY]
        CPotential pp

        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrame cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

        double[::1] f = np.zeros(2*ndim)
        int n_components = cp.n_components

    for i in range(nparticles):
        c_particle_potentials[i] = &(<CPotentialWrapper>(particle_potentials[i].c_instance)).cpotential

    args = <void *>(&c_particle_potentials[0])
    all_w = dop853_helper_save_all(&cp, &cf, <FcnEqDiff> Fwrapper_direct_nbody,
                                   w0, t,
                                   ndim, nparticles, args, ntimes,
                                   atol, rtol, nmax)
    return all_w


class DirectNBody:

    def __init__(self, w0, particle_potentials, external_potential=None,
                 units=None):
        """TODO:

        TODO: could add another option, like in other contexts, for "extra_force"
        to support, e.g., dynamical friction

        Parameters
        ----------
        w0
        partcle_potentials
        external_potential
        units


        """
        if not isinstance(w0, gd.PhaseSpacePosition):
            raise TypeError("Initial conditions `w0` must be a "
                            "gala.dynamics.PhaseSpacePosition object, "
                            "not '{}'".format(w0.__class__.__name__))

        nbodies = w0.shape[0]
        if not nbodies == len(particle_potentials):
            raise ValueError("The number of initial conditions in `w0` must "
                             "match the number of particle potentials passed "
                             "in with `particle_potentials`.")

        # First, figure out how to get units - first place to check is the arg
        if units is None:
            # Next, check the particle potentials
            for pp in particle_potentials:
                if pp is not None:
                    units = pp.units
                    break

        # If units is still none, and external_potential is defined, use that:
        if units is None and external_potential is not None:
            units = external_potential.units

        # Now, if units are still None, raise an error!
        if units is None:
            raise ValueError("Could not determine units from input! You must "
                             "either (1) pass in the unit system with `units`,"
                             "(2) set the units on one of the "
                             "particle_potentials, OR (3) pass in an "
                             "`external_potential` with valid units.")

        # Now that we have the unit system, enforce that all potentials are in
        # that system:
        _particle_potentials = []
        for pp in particle_potentials:
            if pp is None:
                pp = gp.NullPotential(units)
            else:
                pp = pp.replace_units(units, copy=True)
            _particle_potentials.append(pp)

        if external_potential is None:
            external_potential = gp.NullPotential(units)
        else:
            external_potential = external_potential.replace_units(units,
                                                                  copy=True)

        self.w0 = w0
        self.units = units
        self.external_potential = external_potential
        self.particle_potentials = _particle_potentials


    def __repr__(self):
        return "<{} bodies={}>".format(self.__class__.__name__,
                                       self.w0.shape[0])


# def direct_nbody(ic, external_potential=None,
#                  r_soften=None, **time_kwargs):
#
#     if r_scale is None:
#         raise NotImplementedError('For now, you must specify the length scale!')
#
#     if m_scale is None:
#         m_scale = mass.sum()
#
#     if r_soften is not None:
#         # TODO: implement a softening length
#         raise NotImplementedError('')
#
#     usys = get_usys(m_scale, r_scale)
#
#     if external_potential is not None:
#         if (isinstance(external_potential, CCompositePotential) or
#                 hasattr(external_potential, 'keys')):
#
#             ext_pot = CCompositePotential()
#             for k, component in external_potential.items():
#                 pars = dict()
#                 for par in component.parameters:
#                     pars[par] = component.parameters[par].decompose(usys).value
#                 ext_pot[k] = component.__class__(units=usys, **pars)
#
#         else:
#             pot_cls = external_potential.__class__
#             pars = dict()
#             for k in external_potential.parameters:
#                 pars[k] = external_potential.parameters[k].decompose(usys).value
#             ext_pot = pot_cls(units=usys, **pars)
#
#         units = external_potential.units
#
#         if 'units' in time_kwargs:
#             raise ValueError('dude')
#
#     else:
#         ext_pot = NullPotential()
#
#         if 'units' not in time_kwargs:
#             raise ValueError('If no external potential is specified, '
#                              'you must specify a unit system for the '
#                              'initial conditions and time stepping.')
#         units = time_kwargs.pop('units')
#
#     ext_ham = Hamiltonian(ext_pot)
#
#     pos = ic.xyz.decompose(usys).value
#     vel = ic.v_xyz.decompose(usys).value
#     w0 = np.ascontiguousarray(np.vstack((pos, vel)).T)
#     ms = mass.decompose(usys).value
#
#     t = parse_time_specification(units=units, **time_kwargs) * units['time']
#     t = t.decompose(usys).value
#
#     ws = _direct_nbody_dop853(w0, ms, t, ext_ham)
#     pos = np.rollaxis(np.array(ws[..., :3]), axis=2)
#     vel = np.rollaxis(np.array(ws[..., 3:]), axis=2)
#
#     orbits = Orbit(
#         pos=(pos * usys['length']).decompose(units),
#         vel=(vel * usys['length']/usys['time']).decompose(units),
#         t=(t*usys['time']).decompose(units))
#
#     return orbits
