# coding: utf-8

""" Read and write potentials to text (YAML) files. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.units as u
from astropy.utils import isiterable
import numpy as np
from astropy.extern import six
import yaml

from ..units import DimensionlessUnitSystem

__all__ = ['load', 'save']

def _unpack_params(p):
    params = p.copy()
    for key,item in six.iteritems(p):
        if '_unit' in key:
            continue

        if isiterable(item) and not isinstance(item, str):
            params[key] = np.array(item).astype(float)
        else:
            params[key] = float(item)

        if key+'_unit' in params:
            params[key] = params[key] * u.Unit(params[key+'_unit'])
            del params[key+'_unit']

    return params

def _parse_component(component, module):
    # need this here for circular import
    from .. import potential as gala_potential

    try:
        class_name = component['class']
    except KeyError:
        raise KeyError("Potential dictionary must contain a key 'class' for "
                       "specifying the name of the Potential class.")

    if 'units' not in component:
        unitsys = None
    else:
        try:
            unitsys = [u.Unit(unit) for ptype,unit in component['units'].items()]
        except KeyError:
            raise KeyError("Potential dictionary must contain a key 'units' with "
                           "a list of strings specifying the unit system.")

    params = component.get('parameters', {})

    # need to crawl the dictionary structure and unpack quantities
    params = _unpack_params(params)

    if module is None:
        potential = gala_potential
    else:
        potential = module

    try:
        Potential = getattr(potential, class_name)
    except AttributeError: # HACK: this might be bad to assume
        Potential = getattr(gala_potential, class_name)
    return Potential(units=unitsys, **params)

def from_dict(d, module=None):
    """
    Convert a dictionary potential specification into a
    :class:`~gala.potential.PotentialBase` subclass object.

    Parameters
    ----------
    d : dict
        Dictionary specification of a potential.
    module : namespace (optional)

    """
    # need this here for circular import
    from .. import potential as gala_potential

    if module is None:
        potential = gala_potential
    else:
        potential = module

    if 'type' in d and d['type'] == 'composite':
        p = getattr(potential, d['class'])()
        for i,component in enumerate(d['components']):
            c = _parse_component(component, module)
            name = component.get('name', str(i))
            p[name] = c

    elif 'type' in d and d['type'] == 'custom':
        param_groups = dict()
        for i,component in enumerate(d['components']):
            c = _parse_component(component, module)

            try:
                name = component['name']
            except KeyError:
                raise KeyError("For custom potentials, component specification must include "
                               "the component name (e.g., name: 'blah')")

            params = component.get('parameters', {})
            params = _unpack_params(params) # unpack quantities
            param_groups[name] = params
        p = getattr(potential, d['class'])(**param_groups)

    else:
        p = _parse_component(d, module)

    return p

# ----------------------------------------------------------------------------

def _pack_params(p):
    params = p.copy()
    for key,item in six.iteritems(p):
        if hasattr(item,'unit'):
            params[key] = item.value
            params[key+'_unit'] = str(item.unit)

        if hasattr(params[key], 'tolist'): # convert array to list
            params[key] = params[key].tolist()

    return params

def _to_dict_help(potential):
    d = dict()

    d['class'] = potential.__class__.__name__

    if not isinstance(potential.units, DimensionlessUnitSystem):
        d['units'] = dict([(str(ptype),str(unit)) for ptype,unit in potential.units.to_dict().items()])

    if len(potential.parameters) > 0:
        params = _pack_params(potential.parameters)
        d['parameters'] = params

    return d

def to_dict(potential):
    """
    Turn a potential object into a dictionary that fully specifies the
    state of the object.

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        The instantiated :class:`~gala.potential.PotentialBase` object.

    """
    from .. import potential as gp

    if isinstance(potential, gp.CompositePotential):
        d = dict()
        d['class'] = potential.__class__.__name__
        d['components'] = []
        for k,p in potential.items():
            comp_dict = _to_dict_help(p)
            comp_dict['name'] = k
            d['components'].append(comp_dict)

        if potential.__class__.__name__ == 'CompositePotential' or \
           potential.__class__.__name__ == 'CCompositePotential':
            d['type'] = 'composite'
        else:
            d['type'] = 'custom'

    else:
        d = _to_dict_help(potential)

    return d

# ----------------------------------------------------------------------------

def load(f, module=None):
    """
    Read a potential specification file and return a
    :class:`~gala.potential.PotentialBase` object instantiated with parameters
    specified in the spec file.

    Parameters
    ----------
    f : str, file_like
        A block of text, filename, or file-like object to parse and read
        a potential from.
    module : namespace (optional)

    """
    if hasattr(f, 'read'):
        p_dict = yaml.load(f.read())
    else:
        with open(os.path.abspath(f), 'r') as fil:
            p_dict = yaml.load(fil.read())

    return from_dict(p_dict, module=module)

def save(potential, f):
    """
    Write a :class:`~gala.potential.PotentialBase` object out to a text (YAML)
    file.

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        The instantiated :class:`~gala.potential.PotentialBase` object.
    f : str, file_like
        A filename or file-like object to write the input potential object to.

    """
    d = to_dict(potential)

    if hasattr(f, 'write'):
        yaml.dump(d, f, default_flow_style=False)
    else:
        with open(f, 'w') as f2:
            yaml.dump(d, f2, default_flow_style=False)

