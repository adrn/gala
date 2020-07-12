#!/usr/bin/env python
# Licensed under an MIT license - see LICENSE

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os
import sys

from setuptools import setup

from extension_helpers import get_extensions


# First provide helpful messages if contributors try and run legacy commands
# for tests or docs.

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    tox -e test

If you don't already have tox installed, you can install it with:

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest

For more information, see:

  http://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if 'test' in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    tox -e build_docs

If you don't already have tox installed, you can install it with:

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

For more information, see:

  http://docs.astropy.org/en/latest/install.html#builddocs
"""

if 'build_docs' in sys.argv or 'build_sphinx' in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()

# ----------------------------------------------------------------------------
# GSL support
#
from subprocess import check_output, CalledProcessError

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
nogsl = bool(int(os.environ.get('GALA_NOGSL', 0)))
gsl_version = os.environ.get('GALA_GSL_VERSION', None)
gsl_prefix = os.environ.get('GALA_GSL_PREFIX', None)

# Auto-detect whether GSL is installed
if (not nogsl or nogsl is None) and gsl_version is None: # GSL support enabled
    cmd = ['gsl-config', '--version']
    try:
        gsl_version = check_output(cmd, env=env).decode('utf-8')
    except (OSError, CalledProcessError):
        gsl_version = None

if gsl_version is not None:
    gsl_version = gsl_version.strip().split('.')

# If the hacky macros file already exists, read from that what to do.
# This means people experimenting might need to run "git clean" to remove all
# temp. build products if they want to switch between installing with GSL and
# no GSL support.
# if os.path.exists(extra_compile_macros_file):
#     with open(extra_compile_macros_file, "r") as f:
#         line = f.read().strip()

#     if line.endswith('0'):
#         gsl_version = None
#         nogsl = True

print("-" * 79)
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

    if gsl_prefix is None:
        # Now get the gsl install location
        cmd = ['gsl-config', '--prefix']
        try:
            gsl_prefix = check_output(cmd, encoding='utf-8')
        except:
            gsl_prefix = str(check_output(cmd, shell=shell))

    gsl_prefix = os.path.normpath(gsl_prefix.strip())

print("-" * 79)

extensions = get_extensions()
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


setup(use_scm_version={'write_to': os.path.join('gala', 'version.py'),
                       'write_to_template': VERSION_TEMPLATE},
      ext_modules=extensions)
