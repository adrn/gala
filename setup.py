#!/usr/bin/env python
# Licensed under an MIT license - see LICENSE

import os
import sys
import warnings
from collections import defaultdict

from setuptools import Extension, setup

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

extra_compile_macros_file = "src/gala/extra_compile_macros.h"

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
nogsl = bool(int(os.environ.get("GALA_NOGSL", "0")))
gsl_version = os.environ.get("GALA_GSL_VERSION", None)
gsl_prefix = os.environ.get("GALA_GSL_PREFIX", None)

# The EXP installation prefix. This directory should contain 'include' and 'lib' subdirs.
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
        "Gala: Warning: GSL version ({}) is below the minimum required version "
        "(1.16). Installing without GSL support. ".format(".".join(gsl_version))
        + _see_msg
    )
    gsl_version = None

else:
    print(
        "Gala: GSL version {} found, installing with GSL support".format(
            ".".join(gsl_version)
        )
    )

    if gsl_prefix is None:
        # Now get the gsl install location
        cmd = ["gsl-config", "--prefix"]
        try:
            gsl_prefix = check_output(cmd, encoding="utf-8")
        except Exception:
            gsl_prefix = str(check_output(cmd))

    gsl_prefix = os.path.normpath(gsl_prefix.strip())


def pkg_config(pkg: str, *pc_args) -> str:
    """
    pkg_config("eigen3", "--cflags")
    pkg_config("eigen3", "--libs")
    """
    cmd = ["pkg-config", *pc_args, pkg]
    try:
        output = check_output(cmd, encoding="utf-8")
    except Exception:
        try:
            output = str(check_output(cmd))
        except Exception:
            # pkg-config is allowed to fail. For example, a module might
            # set C_INCLUDE_PATH but not expose a pc file.
            warnings.warn(f'"{" ".join(cmd)}" failed for {pkg}.')
            output = ""
    return output.strip()


def get_include_flags(pkg: str) -> list[str]:
    """
    First look at EIGEN3_INCLUDE_DIR environment variable, then
    fall back to "pkg-config --cflags eigen3".
    """
    eigen_incl_dir = os.environ.get(f"{pkg.upper()}_INCLUDE_DIR", None)
    if eigen_incl_dir is None:
        # The cflags from pkg-config might contain multiple flags.
        # Just pass them as extra_compile_args rather than include_dirs.
        eigen_incl_flags = pkg_config(pkg, "--cflags").split()
    else:
        eigen_incl_dir = os.path.normpath(eigen_incl_dir.strip())
        eigen_incl_flags = ["-I" + eigen_incl_dir]
    return eigen_incl_flags


if exp_prefix is None:
    print("Gala: installing without EXP support.")
else:
    if pybind11 is None:
        raise RuntimeError("pybind11 is required to build Gala with EXP support.")

    print(f"Gala: installing with EXP support (GALA_EXP_PREFIX={exp_prefix})")

    extra_incl_flags = []
    for lib in ["eigen3", "hdf5", "mpi"]:
        flags = get_include_flags(lib)
        if flags:
            extra_incl_flags.extend(flags)

# =============================================================================
# Cython extensions
#


def get_all_extensions():
    """All Cython extensions"""
    import numpy as np

    extensions = []
    mac_incl_path = "/usr/include/malloc"

    # Base config shared by many extensions:
    def base_cfg():
        cfg = defaultdict(list)
        cfg["include_dirs"].extend(["src/gala", np.get_include(), mac_incl_path])
        cfg["extra_compile_args"].append("-std=c++17")

        # Some READTHEDOCS hacks - see
        # https://github.com/pyFFTW/pyFFTW/pull/161/files
        # https://github.com/pyFFTW/pyFFTW/pull/162/files
        include_dirs = [os.path.join(sys.prefix, "include")]
        library_dirs = [os.path.join(sys.prefix, "lib")]
        cfg["include_dirs"].extend(include_dirs)
        cfg["library_dirs"].extend(library_dirs)
        return cfg

    # ---- gala._cconfig ----
    cfg = base_cfg()
    cfg["sources"].append("src/gala/cconfig.pyx")
    extensions.append(Extension("gala._cconfig", **cfg))

    # ---- gala.dynamics ----

    #     lyapunov
    cfg = base_cfg()
    cfg["include_dirs"].extend(
        ["src/gala/integrate/cyintegrators", "src/gala/potential"]
    )
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/src/cpotential.cpp",
            "src/gala/potential/hamiltonian/src/chamiltonian.cpp",
            "src/gala/integrate/cyintegrators/dopri/dop853.cpp",
            "src/gala/dynamics/lyapunov/dop853_lyapunov.pyx",
        ]
    )
    extensions.append(Extension("gala.dynamics.lyapunov.dop853_lyapunov", **cfg))

    #     mockstream._coord
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].append("src/gala/dynamics/mockstream/_coord.pyx")
    extensions.append(Extension("gala.dynamics.mockstream._coord", **cfg))

    #     mockstream.df
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].extend(
        [
            "src/gala/dynamics/mockstream/df.pyx",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.dynamics.mockstream.df", **cfg))

    #     mockstream._mockstream
    cfg = base_cfg()
    cfg["include_dirs"].extend(
        [
            "src/gala/integrate/cyintegrators",
            "src/gala/potential",
            "src/gala/dynamics/nbody",
        ]
    )
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/src/cpotential.cpp",
            "src/gala/potential/hamiltonian/src/chamiltonian.cpp",
            "src/gala/dynamics/mockstream/mockstream.pyx",
            "src/gala/integrate/cyintegrators/dopri/dop853.cpp",
        ]
    )
    extensions.append(Extension("gala.dynamics.mockstream._mockstream", **cfg))

    #     nbody
    cfg = base_cfg()
    cfg["include_dirs"].extend(
        [
            "src/gala/integrate/cyintegrators",
            "src/gala/potential",
        ]
    )
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/src/cpotential.cpp",
            "src/gala/potential/hamiltonian/src/chamiltonian.cpp",
            "src/gala/integrate/cyintegrators/dopri/dop853.cpp",
            "src/gala/dynamics/nbody/nbody.pyx",
        ]
    )
    extensions.append(Extension("gala.dynamics.nbody.nbody", **cfg))

    # ===== gala.integrate extensions =====

    #     leapfrog
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala/dynamics/nbody"])
    cfg["sources"].extend(
        [
            "src/gala/integrate/cyintegrators/leapfrog.pyx",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.integrate.cyintegrators.leapfrog", **cfg))

    #     dop853
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].extend(
        [
            "src/gala/potential/hamiltonian/src/chamiltonian.cpp",
            "src/gala/potential/potential/src/cpotential.cpp",
            "src/gala/integrate/cyintegrators/dop853.pyx",
            "src/gala/integrate/cyintegrators/dopri/dop853.cpp",
        ]
    )
    extensions.append(Extension("gala.integrate.cyintegrators.dop853", **cfg))

    #     ruth4
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala/dynamics/nbody"])
    cfg["sources"].extend(
        [
            "src/gala/integrate/cyintegrators/ruth4.pyx",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.integrate.cyintegrators.ruth4", **cfg))

    # ===== gala.potential extensions =====

    #     cpotential
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala"])
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/cpotential.pyx",
            "src/gala/potential/potential/builtin/builtin_potentials.cpp",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.potential.cpotential", **cfg))

    #     ccompositepotential
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala"])
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/ccompositepotential.pyx",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.potential.ccompositepotential", **cfg))

    #     cybuiltin
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala"])
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/builtin/cybuiltin.pyx",
            "src/gala/potential/potential/builtin/builtin_potentials.cpp",
            "src/gala/potential/potential/builtin/multipole.cpp",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.potential.builtin.cybuiltin", **cfg))

    #     cyexp
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala"])
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/builtin/cyexp.pyx",
            "src/gala/potential/potential/builtin/exp_fields.cc",
        ]
    )
    extensions.append(Extension("gala.potential.potential.builtin.cyexp", **cfg))

    #     cytimeinterp
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala/potential", "src/gala"])
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/builtin/cytimeinterp.pyx",
            "src/gala/potential/potential/builtin/time_interp.cpp",
            "src/gala/potential/potential/builtin/time_interp_wrapper.cpp",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.potential.builtin.cytimeinterp", **cfg))

    #     cframe
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].extend(
        [
            "src/gala/potential/frame/cframe.pyx",
            "src/gala/potential/frame/src/cframe.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.frame.cframe", **cfg))

    #     frames
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].extend(
        [
            "src/gala/potential/frame/builtin/frames.pyx",
            "src/gala/potential/frame/builtin/builtin_frames.cpp",
            "src/gala/potential/frame/src/cframe.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.frame.builtin.frames", **cfg))

    #     scf._computecoeff
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].extend(
        [
            "src/gala/potential/scf/computecoeff.pyx",
            "src/gala/potential/scf/src/bfe_helper.cpp",
            "src/gala/potential/scf/src/coeff_helper.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.scf._computecoeff", **cfg))

    #    scf._bfe
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala", "src/gala/potential"])
    cfg["library_dirs"].append(os.path.join(sys.prefix, "lib"))
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/src/cpotential.cpp",
            "src/gala/potential/potential/builtin/builtin_potentials.cpp",
            "src/gala/potential/scf/bfe.pyx",
            "src/gala/potential/scf/src/bfe.cpp",
            "src/gala/potential/scf/src/bfe_helper.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.scf._bfe", **cfg))

    #    scf._bfe_class
    cfg = base_cfg()
    cfg["include_dirs"].extend(["src/gala", "src/gala/potential"])
    cfg["library_dirs"].append(os.path.join(sys.prefix, "lib"))
    cfg["sources"].extend(
        [
            "src/gala/potential/potential/src/cpotential.cpp",
            "src/gala/potential/potential/builtin/builtin_potentials.cpp",
            "src/gala/potential/scf/bfe_class.pyx",
            "src/gala/potential/scf/src/bfe.cpp",
            "src/gala/potential/scf/src/bfe_helper.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.scf._bfe_class", **cfg))

    #     chamiltonian
    cfg = base_cfg()
    cfg["include_dirs"].append("src/gala/potential")
    cfg["sources"].extend(
        [
            "src/gala/potential/hamiltonian/chamiltonian.pyx",
            "src/gala/potential/hamiltonian/src/chamiltonian.cpp",
            "src/gala/potential/potential/src/cpotential.cpp",
        ]
    )
    extensions.append(Extension("gala.potential.hamiltonian.chamiltonian", **cfg))

    return extensions


extensions = get_all_extensions()
extensions_with_flags = []
for ext in extensions:
    # TODO: -Ofast deprecated with clang
    # -march=native may be useful, depending on the architecture
    ext.extra_compile_args.extend(["-Ofast"])
    ext.extra_link_args.extend(["-Ofast"])

    if ("potential.potential" in ext.name or "scf" in ext.name) and (
        gsl_version is not None
    ):
        if "gsl" not in ext.libraries:
            ext.libraries.append("gsl")
            ext.library_dirs.append(os.path.join(gsl_prefix, "lib"))
            ext.include_dirs.append(os.path.join(gsl_prefix, "include"))

        if "gslcblas" not in ext.libraries:
            ext.libraries.append("gslcblas")

    if "cyexp" in ext.name:
        if exp_prefix is not None:
            exp_lib_path = os.path.join(exp_prefix, "lib")
            if not os.path.exists(exp_lib_path):
                msg = (
                    f"No EXP libraries found in {exp_lib_path}. "
                    "Please set GALA_EXP_PREFIX to the directory that contains the 'lib' and 'include' "
                    "subdirectories of your EXP installation."
                )
                raise RuntimeError(msg)

            ext.include_dirs.append(pybind11.get_include())
            if extra_incl_flags is not None:
                ext.extra_compile_args.extend(extra_incl_flags)

            if "exp" not in ext.libraries:
                ext.libraries.extend(
                    (
                        "exputil",
                        "expui",
                        "yaml-cpp",
                    )
                )
                ext.library_dirs.append(exp_lib_path)
                ext.runtime_library_dirs.append(exp_lib_path)
                ext.include_dirs.append(os.path.join(exp_prefix, "include"))
        else:
            # Skip cyexp extension if EXP is not found
            continue

    extensions_with_flags.append(ext)

print("-" * 79)

with open(extra_compile_macros_file, "w", encoding="utf-8") as f:
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
        "write_to": os.path.join("src", "gala", "_version.py"),
        "write_to_template": VERSION_TEMPLATE,
    },
    ext_modules=extensions_with_flags,
)
