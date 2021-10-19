"""Checks for optional dependencies using lazy import from
`PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""
import importlib
from collections.abc import Sequence

# First, the top-level packages:
# TODO: This list is a duplicate of the dependencies in setup.cfg "all", but
# some of the package names are different from the pip-install name (e.g.,
# beautifulsoup4 -> bs4).
_optional_deps = ['h5py', 'sympy', 'tqdm', 'twobody']
_deps = {k.upper(): k for k in _optional_deps}

# Any subpackages that have different import behavior:
_deps['MATPLOTLIB'] = ('matplotlib', 'matplotlib.pyplot')
_deps['GALPY'] = ('galpy', 'galpy.orbit', 'galpy.potential')

__all__ = [f"HAS_{pkg}" for pkg in _deps]


def __getattr__(name):
    if name in __all__:
        module_name = name[4:]
        modules = _deps[module_name]

        if not isinstance(modules, Sequence) or isinstance(modules, str):
            modules = [modules]

        for module in modules:
            try:
                importlib.import_module(module)
            except (ImportError, ModuleNotFoundError):
                return False
            return True

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
