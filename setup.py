# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from distutils.core import setup
from distutils.extension import Extension

# Third-party
import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Get numpy path
numpy_base_path = os.path.split(np.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")
mac_incl_path = "/usr/include/malloc"

extensions = []

potential = Extension("gary.potential.*",
                      ["gary/potential/*.pyx",
                       "gary/potential/_cbuiltin.c"],
                      include_dirs=[numpy_incl_path, mac_incl_path])
extensions.append(potential)

integrate = Extension("gary.integrate.*",
                      ["gary/integrate/*.pyx",
                       "gary/integrate/dopri/dop853.c",
                       "gary/integrate/1d/simpson.c"],
                      include_dirs=[numpy_incl_path, mac_incl_path],
                      extra_compile_args=['-std=c99'])
extensions.append(integrate)

dynamics = Extension("gary.dynamics.*",
                     ["gary/dynamics/*.pyx",
                      "gary/dynamics/brent.c",
                      "gary/integrate/1d/simpson.c"],
                     include_dirs=[numpy_incl_path])
extensions.append(dynamics)

setup(
    name="gary",
    version="0.1",
    author="Adrian M. Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="MIT",
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    packages=["gary", "gary.coordinates", "gary.io",
              "gary.observation", "gary.integrate",
              "gary.dynamics", "gary.inference",
              "gary.potential"],
    scripts=['bin/plotsnap', 'bin/moviesnap', 'bin/snap2gal'],
    package_data={'gary.potential': ['*.pxd','*.c'],
                  'gary.integrate': ['*.pxd','*.c'],
                  'gary.dynamics': ['*.pxd','*.c']
                  },
)
