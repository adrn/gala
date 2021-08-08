"""
Gala.
"""

import sys

__author__ = 'adrn <adrianmpw@gmail.com>'

from ._astropy_init import *

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
__minimum_python_version__ = "3.7"


class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple((int(val)
                             for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError(
        f"packagename does not support Python < {__minimum_python_version__}")
