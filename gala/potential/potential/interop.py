"""Interoperability with other dynamics packages"""

from astropy.constants import G
import astropy.units as u
import numpy as np

import gala.potential as gp
from gala.tests.optional_deps import HAS_GALPY

###############################################################################
# Galpy

if HAS_GALPY:
    import galpy.potential as galpy_gp

    # TODO: some potential conversions drop parameters. Might want to add an
    # option for a custom validator function or something to raise warnings?
    gala_to_galpy = {
        gp.HernquistPotential: (
            galpy_gp.HernquistPotential, {
                'a': 'c',
                'amp': lambda pars, ro, vo: (G*2*pars['m']).to_value(ro*vo**2)
            }
        ),
        gp.IsochronePotential: (
            galpy_gp.IsochronePotential, {
                'b': 'a'
            }
        ),
        gp.JaffePotential: (
            galpy_gp.JaffePotential, {
                'c': 'a'
            }
        ),
        gp.KeplerPotential: (
            galpy_gp.KeplerPotential, {}
        ),
        gp.KuzminPotential: (
            galpy_gp.KuzminDiskPotential, {}
        ),
        gp.LogarithmicPotential: (
            galpy_gp.LogarithmicHaloPotential, {
                'amp': lambda pars, ro, vo: pars['v_c'].to_value(vo)**2,
                'core': 'r_h',
                'q': lambda pars, *_: pars['q3'],
            }
        ),
        gp.LongMuraliBarPotential: (
            galpy_gp.SoftenedNeedleBarPotential, {
                'a': 'a',
                'b': 'b',
                'c': 'c',
                'pa': lambda pars, *_: pars['alpha'].to_value(u.rad)
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
                'a': 'b'
            }
        ),
        gp.PowerLawCutoffPotential: (
            galpy_gp.PowerSphericalPotentialwCutoff, {
                'rc': 'r_c'
            }
        ),
    }


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

    if isinstance(potential, gp.CompositePotential):
        pot = []
        for k in potential.keys():
            pot.append(
                gala_to_galpy_potential(potential[k], ro, vo))

    else:
        if potential.__class__ not in gala_to_galpy:
            raise TypeError(
                f"Converting potential class {potential.__class__.__name__} "
                "to galpy is currently not supported")

        galpy_cls, converters = gala_to_galpy[potential.__class__]
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
                else:
                    # TODO: raise a warning here??
                    galpy_pars[galpy_par_name] = par.value

        pot = galpy_cls(**galpy_pars, ro=ro, vo=vo)

    return pot
