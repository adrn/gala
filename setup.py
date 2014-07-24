# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from distutils.core import setup
from distutils.extension import Extension

# Third-party
import numpy as np
from Cython.Distutils import build_ext

# Get numpy path
# For future potential stuff...
# numpy_base_path = os.path.split(numpy.__file__)[0]
# numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

# lm10_acc = Extension("streams.potential._lm10_acceleration",
#                       ["streams/potential/_lm10_acceleration.pyx"],
#                      include_dirs=[numpy_incl_path])
# pal5_acc = Extension("streams.potential._pal5_acceleration",
#                       ["streams/potential/_pal5_acceleration.pyx"],
#                      include_dirs=[numpy_incl_path])

setup(
    name="streamteam",
    version="0.1",
    author="Adrian M. Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="BSD",
    packages=["streamteam", "streamteam.coordinates", "streamteam.io",
              "streamteam.observation", "streamteam.integrate",
              "streamteam.dynamics", "streamteam.inference",
              "streamteam.potential"]
)
