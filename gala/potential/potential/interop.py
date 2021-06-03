"""Interoperability with other dynamics packages"""

import inspect

from astropy.constants import G
import astropy.units as u
import numpy as np

import gala.potential.potential.builtin as gp
from gala.potential.potential.ccompositepotential import CCompositePotential
from gala.potential.potential.core import CompositePotential
from gala.units import galactic
from gala.tests.optional_deps import HAS_GALPY

__all__ = ['gala_to_galpy_potential', 'galpy_to_gala_potential']

###############################################################################
# Galpy interoperability
#

if HAS_GALPY:
    from scipy.special import gamma
    import galpy.potential as galpy_gp

    def _powerlaw_amp_to_galpy(pars, ro, vo):
        # I don't really remember why this is like this, but it might be related
        # to the difference between GSL gamma and scipy gamma??
        fac = ((1/(2*np.pi) * pars['r_c'].to_value(ro)**(pars['alpha'] - 3) /
               (gamma(3/2 - pars['alpha']/2))))
        amp = fac * (G * pars['m']).to_value(vo**2 * ro)
        return amp

    def _powerlaw_m_from_galpy(pars, ro, vo):
        # See note above!
        fac = ((1/(2*np.pi) * pars['rc']**(pars['alpha'] - 3) /
               (gamma(3/2 - pars['alpha']/2))))
        amp = pars['amp'] * vo**2 * ro
        m = amp / G / fac
        return m

    # TODO: some potential conversions drop parameters. Might want to add an
    # option for a custom validator function or something to raise warnings?
    _gala_to_galpy = {
        gp.HernquistPotential: (
            galpy_gp.HernquistPotential, {
                'a': 'c',
                'amp': lambda pars, ro, vo: (G*2*pars['m']).to_value(ro*vo**2)
            }
        ),
        gp.IsochronePotential: (
            galpy_gp.IsochronePotential, {
                'b': 'b'
            }
        ),
        gp.JaffePotential: (
            galpy_gp.JaffePotential, {
                'a': 'c'
            }
        ),
        gp.KeplerPotential: (galpy_gp.KeplerPotential, {}),
        gp.KuzminPotential: (
            galpy_gp.KuzminDiskPotential, {
                'a': 'a',
            }
        ),
        gp.LogarithmicPotential: (
            galpy_gp.LogarithmicHaloPotential, {
                'amp': lambda pars, ro, vo: pars['v_c'].to_value(vo)**2,
                'core': 'r_h',
                'q': 'q3'
            }
        ),
        gp.LongMuraliBarPotential: (
            galpy_gp.SoftenedNeedleBarPotential, {
                'a': 'a',
                'b': 'b',
                'c': 'c',
                'pa': 'alpha'
            }
        ),
        gp.MiyamotoNagaiPotential: (
            galpy_gp.MiyamotoNagaiPotential, {
                'a': 'a',
                'b': 'b'
            }
        ),
        gp.NFWPotential: (
            galpy_gp.TriaxialNFWPotential, {
                'a': 'r_s',
                'b': lambda pars, *_: pars['b'] / pars['a'],
                'c': lambda pars, *_: pars['c'] / pars['a'],
            }
        ),
        gp.PlummerPotential: (
            galpy_gp.PlummerPotential, {
                'b': 'b'
            }
        ),
        gp.PowerLawCutoffPotential: (
            galpy_gp.PowerSphericalPotentialwCutoff, {
                'amp': _powerlaw_amp_to_galpy,
                'rc': 'r_c',
                'alpha': 'alpha'
            }
        ),
    }

    _galpy_to_gala = {}
    for gala_cls, (galpy_cls, pars) in _gala_to_galpy.items():
        galpy_pars = {v: k for k, v in pars.items()
                      if isinstance(v, (str, int, float, np.ndarray))}
        _galpy_to_gala[galpy_cls] = (gala_cls, galpy_pars)

    # Special cases:
    _galpy_to_gala[galpy_gp.HernquistPotential][1]['m'] = \
        lambda pars, ro, vo: (pars['amp'] * ro * vo**2 / G / 2)

    _galpy_to_gala[galpy_gp.LogarithmicHaloPotential][1]['v_c'] = \
        lambda pars, ro, vo: np.sqrt(pars['amp'] * vo**2)

    _galpy_to_gala[galpy_gp.TriaxialNFWPotential][1]['m'] = \
        lambda pars, ro, vo: (
            pars['amp'] * ro * vo**2 / G * 4*np.pi*pars['a']**3)
    _galpy_to_gala[galpy_gp.TriaxialNFWPotential][1]['a'] = 1.
    _galpy_to_gala[galpy_gp.TriaxialNFWPotential][1]['b'] = 'b'
    _galpy_to_gala[galpy_gp.TriaxialNFWPotential][1]['c'] = 'c'

    _galpy_to_gala[galpy_gp.PowerSphericalPotentialwCutoff][1]['m'] = \
        _powerlaw_m_from_galpy

    _galpy_to_gala[galpy_gp.NFWPotential] = (
        gp.NFWPotential, {
            'r_s': 'a',
        }
    )


def _get_ro_vo(ro, vo):
    # If not specified, get the default ro, vo from Galpy
    if ro is None or vo is None:
        from galpy.potential import Force
        f = Force()

        if ro is None:
            ro = f._ro * u.kpc
        if vo is None:
            vo = f._vo * u.km/u.s

    return u.Quantity(ro), u.Quantity(vo)


def gala_to_galpy_potential(potential, ro=None, vo=None):

    if not HAS_GALPY:
        raise ImportError(
            "Failed to import galpy.potential: Converting a potential to a "
            "galpy potential requires galpy to be installed.")

    ro, vo = _get_ro_vo(ro, vo)

    if isinstance(potential, CompositePotential):
        pot = []
        for k in potential.keys():
            pot.append(
                gala_to_galpy_potential(potential[k], ro, vo))

    else:
        if potential.__class__ not in _gala_to_galpy:
            raise TypeError(
                f"Converting potential class {potential.__class__.__name__} "
                "to galpy is currently not supported")

        galpy_cls, converters = _gala_to_galpy[potential.__class__]
        gala_pars = potential.parameters.copy()

        galpy_pars = {}
        if 'amp' not in converters and 'm' not in gala_pars:
            raise ValueError("Gala potential has no mass parameter, so "
                             "converting to a Galpy potential is currently "
                             "not supported.")

        converters.setdefault(
            'amp', lambda pars, ro, vo: (G * pars['m']).to_value(ro * vo**2))

        for galpy_par_name, conv in converters.items():
            if isinstance(conv, str):
                galpy_pars[galpy_par_name] = gala_pars[conv]
            elif hasattr(conv, '__call__'):
                galpy_pars[galpy_par_name] = conv(gala_pars, ro, vo)
            elif isinstance(conv, (int, float, u.Quantity, np.ndarray)):
                galpy_pars[galpy_par_name] = conv
            else:
                # TODO: invalid parameter??
                print(f"FAIL: {galpy_par_name}, {conv}")

            par = galpy_pars[galpy_par_name]
            if hasattr(par, 'unit'):
                if par.unit.physical_type == 'length':
                    galpy_pars[galpy_par_name] = par.to_value(ro)
                elif par.unit.physical_type == 'dimensionless':
                    galpy_pars[galpy_par_name] = par.value
                elif par.unit.physical_type == 'angle':
                    galpy_pars[galpy_par_name] = par.to_value(u.rad)
                else:
                    # TODO: raise a warning here??
                    galpy_pars[galpy_par_name] = par.value

        pot = galpy_cls(**galpy_pars, ro=ro, vo=vo)

    return pot


def galpy_to_gala_potential(potential, ro=None, vo=None, units=galactic):

    if not HAS_GALPY:
        raise ImportError(
            "Failed to import galpy.potential: Converting a potential to a "
            "gala potential requires galpy to be installed.")

    ro, vo = _get_ro_vo(ro, vo)

    if potential._roSet:
        ro = potential._ro * u.kpc
    if potential._voSet:
        vo = potential._vo * u.km/u.s

    if isinstance(potential, list):
        pot = CCompositePotential()
        for i, sub_pot in enumerate(potential):
            pot[str(i)] = galpy_to_gala_potential(sub_pot, ro, vo)

    else:
        if potential.__class__ not in _galpy_to_gala:
            raise TypeError(
                f"Converting galpy potential {potential.__class__.__name__} "
                "to gala is currently not supported")

        gala_cls, converters = _galpy_to_gala[potential.__class__]

        exclude = ['self', 'normalize', 'ro', 'vo']
        spec = inspect.getfullargspec(potential.__class__)
        par_names = [arg for arg in spec.args if arg not in exclude]

        # UGH!
        galpy_pars = {}
        for name in par_names:
            galpy_pars[name] = getattr(potential,
                                       '_' + name,
                                       getattr(potential, name, None))

        if isinstance(potential, galpy_gp.LogarithmicHaloPotential):
            galpy_pars['core'] = np.sqrt(potential._core2)

        elif isinstance(potential, galpy_gp.SoftenedNeedleBarPotential):
            galpy_pars['c'] = np.sqrt(potential._c2)

        if 'm' in inspect.getfullargspec(gala_cls).args:
            converters.setdefault(
                'm', lambda pars, ro, vo: pars['amp'] * ro * vo**2 / G
            )

        gala_pars = {}
        for gala_par_name, conv in converters.items():
            if isinstance(conv, str):
                gala_pars[gala_par_name] = galpy_pars[conv]
            elif hasattr(conv, '__call__'):
                gala_pars[gala_par_name] = conv(galpy_pars, ro, vo)
            elif isinstance(conv, (int, float, u.Quantity, np.ndarray)):
                gala_pars[gala_par_name] = conv
            else:
                # TODO: invalid parameter??
                print(f"FAIL: {gala_par_name}, {conv}")

            if hasattr(gala_pars[gala_par_name], 'unit'):
                continue

            gala_par = gala_cls._parameters[gala_par_name]
            if gala_par.physical_type == 'mass':
                gala_pars[gala_par_name] = gala_pars[gala_par_name] * u.Msun
            elif gala_par.physical_type == 'length':
                gala_pars[gala_par_name] = gala_pars[gala_par_name] * ro
            elif gala_par.physical_type == 'speed':
                gala_pars[gala_par_name] = gala_pars[gala_par_name] * vo
            elif gala_par.physical_type == 'angle':
                gala_pars[gala_par_name] = gala_pars[gala_par_name] * u.radian
            elif gala_par.physical_type == 'dimensionless':
                pass
            else:
                print("TODO")

        pot = gala_cls(**gala_pars, units=units)

    return pot
