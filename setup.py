#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys
from subprocess import check_output, CalledProcessError

import builtins

# Ensure that astropy-helpers is available
import ah_bootstrap  # noqa

from setuptools import setup
from setuptools.config import read_configuration

from astropy_helpers.setup_helpers import (register_commands, get_package_info,
                                           get_distutils_build_option)
from astropy_helpers.version_helpers import generate_version_py

# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = read_configuration('setup.cfg')['metadata']['name']

# Create a dictionary with setup command overrides. Note that this gets
# information about the package (name and version) from the setup.cfg file.
cmdclass = register_commands()

# Freeze build information in version.py. Note that this gets information
# about the package (name and version) from the setup.cfg file.
version = generate_version_py()

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

# ----------------------------------------------------------------------------
# GSL support
#
extra_compile_macros_file = 'gala/extra_compile_macros.h'

# Note: on RTD, they now support conda environments, but don't activate the
# conda environment that gets created, and so the C stuff installed with GSL
# aren't picked up. This is my attempt to hack around that!
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    PATH = '/home/docs/checkouts/readthedocs.org/user_builds/gala-astro/conda/latest/bin/'
    env = os.environ.copy()
    env['PATH'] = env.get('PATH', "") + ":" + PATH
else:
    env = None

# First, see if the user wants to install without GSL:
nogsl = get_distutils_build_option('nogsl')

# Auto-detect whether GSL is installed
if not nogsl or nogsl is None: # GSL support enabled
    cmd = ['gsl-config', '--version']
    try:
        gsl_version = check_output(cmd, env=env)
    except (OSError, CalledProcessError):
        gsl_version = None
    else:
        gsl_version = gsl_version.decode('utf-8').strip().split('.')

else:
    gsl_version = None

# If the hacky macros file already exists, read from that what to do.
# This means people experimenting might need to run "git clean" to remove all
# temp. build products if they want to switch between installing with GSL and
# no GSL support.
if os.path.exists(extra_compile_macros_file):
    with open(extra_compile_macros_file, "r") as f:
        line = f.read().strip()

    if line.endswith('0'):
        gsl_version = None
        nogsl = True

print()
_see_msg = ("See the gala documentation 'installation' page for more "
            "information about GSL support and installing GSL: "
            "http://gala.adrian.pw/en/latest/install.html")
if gsl_version is None:
    if nogsl:
        print('Installing without GSL support.')
    else:
        print('GSL not found: installing without GSL support. ' + _see_msg)

elif gsl_version < ['1', '14']:
    print('Warning: GSL version ({0}) is below the minimum required version '
          '(1.16). Installing without GSL support. '
          .format('.'.join(gsl_version)) + _see_msg)
    gsl_version = None

else:
    print("GSL version {0} found: installing with GSL support"
          .format('.'.join(gsl_version)))

    # Now get the gsl install location
    cmd = ['gsl-config', '--prefix']
    try:
        gsl_prefix = check_output(cmd, encoding='utf-8').strip()
    except:
        gsl_prefix = str(check_output(cmd)).strip()

print()

extensions = package_info['ext_modules']
for ext in extensions:
    if 'potential.potential' in ext.name or 'scf' in ext.name:
        if gsl_version is not None:
            if 'gsl' not in ext.libraries:
                ext.libraries.append('gsl')
                ext.library_dirs.append(os.path.join(gsl_prefix, 'lib'))
                ext.include_dirs.append(os.path.join(gsl_prefix, 'include'))

            if 'gslcblas' not in ext.libraries:
                ext.libraries.append('gslcblas')

with open(extra_compile_macros_file, 'w') as f:
    if gsl_version is not None:
        f.writelines(['#define USE_GSL 1'])
    else:
        f.writelines(['#define USE_GSL 0'])

setup(name='astro-gala', version=version, cmdclass=cmdclass, **package_info)
