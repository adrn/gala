#!/usr/bin/env python
# Licensed under an MIT license - see LICENSE

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os
import sys

from extension_helpers import get_extensions
from setuptools import setup

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

if "test" in sys.argv:
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

if "build_docs" in sys.argv or "build_sphinx" in sys.argv:
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
from subprocess import CalledProcessError, check_output

extra_compile_macros_file = "gala/extra_compile_macros.h"

# Note: on RTD, they now support conda environments, but don't activate the
# conda environment that gets created, and so the C stuff installed with GSL
# aren't picked up. This is my attempt to hack around that!
on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    PATH = (
        "/home/docs/checkouts/readthedocs.org/user_builds/gala-astro/conda/latest/bin/"
    )
    env = os.environ.copy()
    env["PATH"] = env.get("PATH", "") + ":" + PATH
else:
    env = None

# First, see if the user wants to install without GSL:
nogsl = bool(int(os.environ.get("GALA_NOGSL", 0)))
gsl_version = os.environ.get("GALA_GSL_VERSION", None)
gsl_prefix = os.environ.get("GALA_GSL_PREFIX", None)

exp_prefix = os.environ.get("GALA_EXP_PREFIX", None)

try:
    import pybind11
except ImportError:
    pybind11 = None


# Auto-detect whether GSL is installed
if (not nogsl or nogsl is None) and gsl_version is None:  # GSL support enabled
    cmd = ["gsl-config", "--version"]
    try:
        gsl_version = check_output(cmd, env=env).decode("utf-8")
    except (OSError, CalledProcessError):
        gsl_version = None

if gsl_version is not None:
    gsl_version = gsl_version.strip().split(".")

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
_see_msg = (
    "See the gala documentation 'installation' page for more "
    "information about GSL support and installing GSL: "
    "http://gala.adrian.pw/en/latest/install.html"
)
if gsl_version is None:
    if nogsl:
        print("Gala: Installing without GSL support.")
    else:
        print("Gala: GSL not found, installing without GSL support. " + _see_msg)

elif gsl_version < ["1", "14"]:
    print(
        "Gala: Warning: GSL version ({0}) is below the minimum required version "
        "(1.16). Installing without GSL support. ".format(".".join(gsl_version))
        + _see_msg
    )
    gsl_version = None

else:
    print(
        "Gala: GSL version {0} found, installing with GSL support".format(
            ".".join(gsl_version)
        )
    )

    if gsl_prefix is None:
        # Now get the gsl install location
        cmd = ["gsl-config", "--prefix"]
        try:
            gsl_prefix = check_output(cmd, encoding="utf-8")
        except:
            gsl_prefix = str(check_output(cmd))

    gsl_prefix = os.path.normpath(gsl_prefix.strip())


eigen_incl = os.environ.get("EIGEN3_INCLUDE_DIR", None)
if exp_prefix is None:
    print("Gala: installing without EXP support.")
else:
    if pybind11 is None:
        raise RuntimeError("pybind11 is required to build Gala with EXP support.")

    print(f"Gala: installing with EXP support (GALA_EXP_PREFIX={exp_prefix})")

    if eigen_incl is None:
        cmd = ["pkg-config", "--cflags", "eigen3"]
        try:
            eigen_incl = check_output(cmd, encoding="utf-8")
        except:
            eigen_incl = str(check_output(cmd))
        eigen_incl = eigen_incl[2:]

    eigen_incl = os.path.normpath(eigen_incl.strip())

print("-" * 79)

all_extensions = get_extensions()
extensions = []
for ext in all_extensions:
    if "potential.potential" in ext.name or "scf" in ext.name:
        if gsl_version is not None:
            if "gsl" not in ext.libraries:
                ext.libraries.append("gsl")
                ext.library_dirs.append(os.path.join(gsl_prefix, "lib"))
                ext.include_dirs.append(os.path.join(gsl_prefix, "include"))

            if "gslcblas" not in ext.libraries:
                ext.libraries.append("gslcblas")

    if "cyexp" in ext.name:
        if exp_prefix is not None:
            ext.include_dirs.append(pybind11.get_include())
            if eigen_incl is not None:
                ext.include_dirs.append(eigen_incl)

            if "exp" not in ext.libraries:
                # TODO: we're compiling against installed EXP libraries,
                # but headers from the source, because EXP doesn't install
                # its headers. It would also need to install its vendored
                # headers.

                ext.libraries.extend(
                    (
                        "exputil",
                        "expui",
                        "yaml-cpp",
                    )
                )
                #TODO: this requires user to install EXP to $GALA_EXP_PREFIX/install
                exp_lib = os.path.join(exp_prefix, "install", "lib")

                ext.library_dirs.append(exp_lib)
                ext.runtime_library_dirs.append(exp_lib)
                ext.include_dirs.extend(
                    (
                        os.path.join(exp_prefix, "include"),
                        # TODO: requires build in $GALA_EXP_PREFIX/build
                        os.path.join(exp_prefix, "build"),
                        os.path.join(exp_prefix, "expui"),
                        os.path.join(exp_prefix, "extern", "HighFive", "include"),
                        os.path.join(exp_prefix, "extern", "yaml-cpp", "include"),
                    )
                )

        else:
            # Skip cyexp extension if EXP is not found
            continue

    extensions.append(ext)

with open(extra_compile_macros_file, "w") as f:
    if gsl_version is not None:
        f.write("#define USE_GSL 1\n")
    else:
        f.write("#define USE_GSL 0\n")

    if exp_prefix is not None:
        f.write("#define USE_EXP 1\n")
    else:
        f.write("#define USE_EXP 0\n")


setup(
    use_scm_version={
        "write_to": os.path.join("gala", "_version.py"),
        "write_to_template": VERSION_TEMPLATE,
    },
    ext_modules=extensions,
)
