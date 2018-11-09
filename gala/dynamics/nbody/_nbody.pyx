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
    ctypedef struct CPotential:
        int n_components
        double *parameters[16];

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrame *fr, unsigned norbits) nogil
    void Fwrapper_direct_nbody(unsigned ndim, double t, double *w, double *f,
                               CPotential *p, CFrame *fr, unsigned norbits)


cpdef _direct_nbody_dop853(double [:, ::1] w0, double [::1] m, double[::1] t,
                           hamiltonian,
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
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]
        unsigned ntimes = len(t)

        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrame cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

        double[::1] mass = m.copy()
        double[::1] f = np.zeros(2*ndim)
        int n_components = cp.n_components

    cp.parameters[n_components] = &mass[0]

    all_w = dop853_helper_save_all(&cp, &cf, <FcnEqDiff> Fwrapper_direct_nbody,
                                   w0, t,
                                   ndim, norbits, ntimes,
                                   atol, rtol, nmax)
    return all_w

    # Fwrapper_direct_nbody(2*ndim, 0.,&w0[0,0], &f[0],
    #                       &cp, &cf, 2)


def get_usys(m_scale, r_scale):
    l_unit = u.Unit(r_scale.to(u.kpc))
    m_unit = u.Unit(m_scale.to(u.Msun))
    t_unit = u.Unit(np.sqrt((l_unit**3) / (G*m_unit)).to(u.Myr))
    a_unit = u.radian
    return UnitSystem(l_unit, m_unit, t_unit, a_unit)


def direct_nbody(ic, mass, external_potential=None,
                 m_scale=None, r_scale=None,
                 r_soften=None, **time_kwargs):

    if r_scale is None:
        raise NotImplementedError('For now, you must specify the length scale!')

    if m_scale is None:
        m_scale = mass.sum()

    if r_soften is not None:
        # TODO: implement a softening length
        raise NotImplementedError('')

    usys = get_usys(m_scale, r_scale)

    if external_potential is not None:
        if (isinstance(external_potential, CCompositePotential) or
                hasattr(external_potential, 'keys')):

            ext_pot = CCompositePotential()
            for k, component in external_potential.items():
                pars = dict()
                for par in component.parameters:
                    pars[par] = component.parameters[par].decompose(usys).value
                ext_pot[k] = component.__class__(units=usys, **pars)

        else:
            pot_cls = external_potential.__class__
            pars = dict()
            for k in external_potential.parameters:
                pars[k] = external_potential.parameters[k].decompose(usys).value
            ext_pot = pot_cls(units=usys, **pars)

        units = external_potential.units

        if 'units' in time_kwargs:
            raise ValueError('dude')

    else:
        ext_pot = NullPotential()

        if 'units' not in time_kwargs:
            raise ValueError('If no external potential is specified, '
                             'you must specify a unit system for the '
                             'initial conditions and time stepping.')
        units = time_kwargs.pop('units')

    ext_ham = Hamiltonian(ext_pot)

    pos = ic.xyz.decompose(usys).value
    vel = ic.v_xyz.decompose(usys).value
    w0 = np.ascontiguousarray(np.vstack((pos, vel)).T)
    ms = mass.decompose(usys).value

    t = parse_time_specification(units=units, **time_kwargs) * units['time']
    t = t.decompose(usys).value

    ws = _direct_nbody_dop853(w0, ms, t, ext_ham)
    pos = np.rollaxis(np.array(ws[..., :3]), axis=2)
    vel = np.rollaxis(np.array(ws[..., 3:]), axis=2)

    orbits = Orbit(
        pos=(pos * usys['length']).decompose(units),
        vel=(vel * usys['length']/usys['time']).decompose(units),
        t=(t*usys['time']).decompose(units))

    return orbits
