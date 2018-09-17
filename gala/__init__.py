"""
Gala.
"""

import sys

__author__ = 'adrn <adrianmpw@gmail.com>'

from ._astropy_init import *

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
__minimum_python_version__ = "3.5"

class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("packagename does not support Python < {}".format(__minimum_python_version__))


# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import coordinates
    from . import dynamics
    from . import integrate
    from . import potential
    from . import units
    from . import util
    # from . import mpl_style
