"""Checks for optional dependencies using lazy import from
`PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""
import importlib

# First, the top-level packages:
# TODO: This list is a duplicate of the dependencies in setup.cfg "all", but
# some of the package names are different from the pip-install name (e.g.,
# beautifulsoup4 -> bs4).
_optional_deps = ['h5py', 'galpy', 'matplotlib', 'sympy']
_deps = {k.upper(): k for k in _optional_deps}

# Any subpackages that have different import behavior:
_deps['PLT'] = 'matplotlib.pyplot'
_deps['GALPY_ORBIT'] = 'galpy.orbit'
_deps['GALPY_POTENTIAL'] = 'galpy.potential'

__all__ = [f"HAS_{pkg}" for pkg in _deps]


def __getattr__(name):
    if name in __all__:
        module_name = name[4:]

        try:
            importlib.import_module(_deps[module_name])
        except (ImportError, ModuleNotFoundError):
            return False
        return True

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
