.. include:: references.txt

.. _gala-install:

************
Installation
************

With ``uv`` and ``pip`` (recommended)
=====================================

To install the latest stable version using ``uv pip``, use::

    uv pip install gala

This is the recommended way to install ``gala``.

To install the development version::

    uv pip install git+https://github.com/adrn/gala

Or, to add ``gala`` to an existing ``uv`` environment::

    uv add gala

From Source: Cloning, Building, Installing
==========================================

The latest development version of gala can be cloned from
`GitHub <https://github.com/>`_ using ``git``::

    git clone git://github.com/adrn/gala.git

To build and install the project (from the root of the source tree, e.g., inside
the cloned ``gala`` directory)::

    uv pip install .


Architecture-Specific Optimizations
===================================

For performance reasons, the pre-compiled wheels installed via ``pip`` are built
assuming a minimum CPU architecture. For x86-64 CPUs (e.g. Intel), the wheels are built
against ``x86-64-v3``, which is supported by most Intel CPUs since 2013. For MacOS on
ARM, the wheels are built against ``apple-m1``, which should work on Apple M1 (2020) or
newer.

For the best performance, you may wish to build from source (see above) with the
following environment variable set::

    export CXXFLAGS=-march=native

This will likely have the biggest effect on orbit integration. Be aware that compiling
with this flag means that Gala will only run on the same type of CPU that it was
compiled on!

If your CPU does not support the instruction set that Gala was compiled for, you will
likely receive an "illegal instruction error" (``SIGILL``). If that happens, try
recompiling from source, without any ``-march`` flags.


Installing on Windows
=====================

We have successfully installed Gala on Windows within an Anaconda installation, or with
the Windows Subsystem for Linux (WSL), which acts as a Linux environment within Windows.
Either way, we recommend using GCC to compile any C code. Unfortunately, Gala will not
work with Microsoft Visual Studio's C compiler because it is not C99 compliant.


GSL support
===========

Some functionality in Gala depends on the GNU Scientific Library (GSL), a C
library for numerical and mathematical programming. By default, Gala will
determine whether to install with or without GSL support depending on whether it
can find a GSL installation on your machine. If you are not sure whether you
have GSL installed or not, try running::

    gsl-config --version

in your terminal. If that returns a version number, you likely have GSL
installed. If it errors, you will need to install it. Additionally, if your
version of GSL is <1.14, we recommend updating to a newer version, as Gala has
only been tested with GSL >= 1.14.

On Linux and Mac, you can install GSL using a package manager, such as ``apt`` or
``homebrew``. For example, on a Mac with ``homebrew``, you can install GSL with::

    brew install gsl

Or on Linux with ``apt``::

    apt-get install gsl-bin libgsl0-dev


Forcing gala to install without GSL support
-------------------------------------------

You can force Gala to build without GSL support using the ``--nogsl`` flag passed to
setup.py. To use this flag, you must install Gala from source by cloning the repository
(see above) and running::

    uv pip install gala --install-option="--nogsl"


Python Dependencies
===================

Gala has the following build dependencies:

* `Python`_ >= 3.11
* `Numpy`_
* `Cython <http://www.cython.org/>`_
* ``setuptools``
* ``setuptools_scm``
* ``pybind11``

Gala has the following runtime dependencies:

* `Numpy`_
* `Astropy`_
* `PyYAML`_
* `scipy`_


Optional
--------

- `Sympy`_ for creating :class:`~gala.potential.potential.PotentialBase`
  subclass instances from a mathematical expression using
  :func:`~gala.potential.potential.from_equation()`.
- ``galpy``
- ``h5py``
- ``matplotlib``
