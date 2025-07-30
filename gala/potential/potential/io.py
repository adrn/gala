"""Read and write potentials to text (YAML) files."""

import os

import astropy.units as u
import numpy as np
import yaml
from astropy.utils import isiterable

from gala.units import DimensionlessUnitSystem

__all__ = ["load", "save"]


def _unpack_params(p):
    params = p.copy()
    for key, item in p.items():
        if "_unit" in key:
            continue

        if isiterable(item) and not isinstance(item, str):
            params[key] = np.array(item).astype(float)
        else:
            params[key] = float(item)

        if key + "_unit" in params:
            params[key] *= u.Unit(params[key + "_unit"])
            del params[key + "_unit"]

    return params


def _parse_component(component, module):
    # need this here for circular import
    from .. import potential as gala_potential

    try:
        class_name = component["class"]
    except KeyError as e:
        raise KeyError(
            "Potential dictionary must contain a key 'class' for "
            "specifying the name of the Potential class."
        ) from e

    if "units" not in component:
        unitsys = None
    else:
        try:
            unitsys = [u.Unit(unit) for ptype, unit in component["units"].items()]
        except KeyError as e:
            raise KeyError(
                "Potential dictionary must contain a key 'units' "
                "with a list of strings specifying the unit system."
            ) from e

    params = component.get("parameters", {})

    # need to crawl the dictionary structure and unpack quantities
    params = _unpack_params(params)

    potential = gala_potential if module is None else module

    try:
        Potential = getattr(potential, class_name)
    except AttributeError:  # HACK: this might be bad to assume
        Potential = getattr(gala_potential, class_name)

    return Potential(units=unitsys, **params)


def from_dict(d, module=None):
    """
    Convert a dictionary potential specification into a potential object.

    This function parses a dictionary representation of a potential and
    creates the corresponding :class:`~gala.potential.PotentialBase`
    subclass instance. Supports both simple potentials and composite
    potentials with multiple components.

    Parameters
    ----------
    d : dict
        Dictionary specification of a potential. Must contain at minimum
        a 'class' key specifying the potential class name. For composite
        potentials, should include 'type': 'composite' and a 'components'
        list of component dictionaries.
    module : module, optional
        Python module namespace to search for potential classes. If not
        provided, uses `gala.potential`.

    Returns
    -------
    potential : `~gala.potential.PotentialBase`
        The instantiated potential object.

    Examples
    --------
    Create a simple Hernquist potential::

        >>> pot_dict = {'class': 'HernquistPotential', 'm': 1e11, 'c': 2.0}
        >>> pot = from_dict(pot_dict)

    Create a composite potential::

        >>> comp_dict = {
        ...     'type': 'composite',
        ...     'class': 'CompositePotential',
        ...     'components': [
        ...         {'class': 'HernquistPotential', 'm': 1e10, 'c': 1.0},
        ...         {'class': 'NFWPotential', 'm': 1e12, 'r_s': 20.0}
        ...     ]
        ... }
        >>> comp_pot = from_dict(comp_dict)
    """

    # need this here for circular import issues
    import gala.potential as gala_potential

    potential = gala_potential if module is None else module

    if "type" in d and d["type"] == "composite":
        p = getattr(potential, d["class"])()
        for i, component in enumerate(d["components"]):
            c = _parse_component(component, module)
            name = component.get("name", str(i))
            p[name] = c

    elif "type" in d and d["type"] == "custom":
        param_groups = {}
        for component in d["components"]:
            c = _parse_component(component, module)

            try:
                name = component["name"]
            except KeyError as e:
                raise KeyError(
                    "For custom potentials, component specification "
                    "must include the component name (e.g., name: "
                    "'blah')"
                ) from e

            params = component.get("parameters", {})
            params = _unpack_params(params)  # unpack quantities
            param_groups[name] = params
        p = getattr(potential, d["class"])(**param_groups)

    else:
        p = _parse_component(d, module)

    return p


# ----------------------------------------------------------------------------


def _pack_params(p):
    params = p.copy()
    for key, item in p.items():
        if hasattr(item, "unit"):
            params[key] = item.value
            params[key + "_unit"] = str(item.unit)

        if hasattr(params[key], "tolist"):  # convert array to list
            params[key] = params[key].tolist()

    return params


def _to_dict_help(potential):
    d = {}

    d["class"] = potential.__class__.__name__

    if not isinstance(potential.units, DimensionlessUnitSystem):
        d["units"] = {str(k): str(v) for k, v in potential.units.to_dict().items()}

    if len(potential.parameters) > 0:
        params = _pack_params(potential.parameters)
        d["parameters"] = params

    return d


def to_dict(potential):
    """
    Convert a potential object into a dictionary representation.

    This function serializes a :class:`~gala.potential.PotentialBase`
    object into a dictionary that fully specifies the potential's state,
    including all parameters, units, and structure. The resulting
    dictionary can be used to recreate the potential using
    :func:`~gala.potential.io.from_dict`.

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        The instantiated potential object to convert to dictionary form.

    Returns
    -------
    pot_dict : dict
        Dictionary representation of the potential containing the class
        name, parameters, units, and (for composite potentials) component
        structure.

    Examples
    --------
    Convert a simple potential to dictionary::

        >>> pot = HernquistPotential(m=1e11*u.Msun, c=2*u.kpc)
        >>> pot_dict = to_dict(pot)

    Convert a composite potential::

        >>> comp_pot = CompositePotential(bulge=hernquist, halo=nfw)
        >>> comp_dict = to_dict(comp_pot)

    See Also
    --------
    from_dict : Create potential object from dictionary representation.
    save : Save potential to file.
    """
    from .. import potential as gp

    if isinstance(potential, gp.CompositePotential):
        d = {}
        d["class"] = potential.__class__.__name__
        d["components"] = []
        for k, p in potential.items():
            comp_dict = _to_dict_help(p)
            comp_dict["name"] = k
            d["components"].append(comp_dict)

        if potential.__class__.__name__ in {
            "CompositePotential",
            "CCompositePotential",
        }:
            d["type"] = "composite"
        else:
            d["type"] = "custom"

    else:
        d = _to_dict_help(potential)

    return d


# ----------------------------------------------------------------------------


def load(f, module=None):
    """
    Load a potential from a YAML specification file.

    This function reads a YAML file containing a potential specification
    and creates the corresponding :class:`~gala.potential.PotentialBase`
    object. The file format should match the dictionary structure expected
    by :func:`~gala.potential.io.from_dict`.

    Parameters
    ----------
    f : str, file-like
        Path to a YAML file, a block of YAML text, or a file-like object
        containing the potential specification to parse and load.
    module : module, optional
        Python module namespace to search for potential classes. If not
        provided, uses `gala.potential`.

    Returns
    -------
    potential : :class:`~gala.potential.PotentialBase`
        The loaded potential object.

    Examples
    --------
    Load a potential from a YAML file::

        >>> pot = load('my_potential.yml')

    Load from a YAML string::

        >>> yaml_spec = '''
        ... class: HernquistPotential
        ... parameters:
        ...   m: 1.0e11
        ...   c: 2.0
        ... '''
        >>> pot = load(yaml_spec)

    See Also
    --------
    save : Save potential to YAML file.
    from_dict : Create potential from dictionary specification.
    """
    if hasattr(f, "read"):
        p_dict = yaml.load(f.read(), Loader=yaml.Loader)
    else:
        with open(os.path.abspath(f), encoding="utf-8") as fil:
            p_dict = yaml.load(fil.read(), Loader=yaml.Loader)

    return from_dict(p_dict, module=module)


def save(potential, f):
    """
    Save a potential object to a YAML file.

    This function serializes a :class:`~gala.potential.PotentialBase`
    object to YAML format and writes it to a file. The resulting file
    can be loaded using :func:`~gala.potential.io.load`.

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        The potential object to save.
    f : str, file-like
        Output filename or file-like object to write the potential
        specification to.

    Examples
    --------
    Save a potential to file::

        >>> pot = HernquistPotential(m=1e11*u.Msun, c=2*u.kpc)
        >>> save(pot, 'hernquist_potential.yml')

    Save to a string buffer::

        >>> from io import StringIO
        >>> buffer = StringIO()
        >>> save(pot, buffer)
        >>> yaml_content = buffer.getvalue()

    See Also
    --------
    load : Load potential from YAML file.
    to_dict : Convert potential to dictionary representation.
    """
    d = to_dict(potential)

    if hasattr(f, "write"):
        yaml.dump(d, f, default_flow_style=None)
    else:
        with open(f, "w", encoding="utf-8") as f2:
            yaml.dump(d, f2, default_flow_style=None)
