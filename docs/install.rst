.. include:: references.txt

.. _gala-install:

************
Installation
************

With ``pip`` (recommended)
==========================

To install the latest stable version using ``pip``, use::

    python -m pip install gala

This is the recommended way to install ``gala``.

To install the development version::

    python -m pip install git+https://github.com/adrn/gala


With ``conda``
==============

To install the latest stable version with ``conda``, use the ``conda-forge``
channel with::

    conda install -c conda-forge gala


From Source: Cloning, Building, Installing
==========================================

The latest development version of gala can be cloned from
`GitHub <https://github.com/>`_ using ``git``::

    git clone git://github.com/adrn/gala.git

To build and install the project (from the root of the source tree, e.g., inside
the cloned ``gala`` directory)::

    python -m pip install .


Installing on Windows
=====================

We have successfully installed Gala on Windows within an Anaconda installation,
which installs and uses GCC to compile C code. Unfortunately, Gala will not work
with Microsoft Visual Studio's C compiler because it is not C99 compliant. With
Anaconda, you can install ``gsl`` (see below) and then install Gala with
``pip``::

    pip install gala


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

Installing with ``conda``
-------------------------

If you use a Mac computer, we recommend installing GSL using the `anaconda
<https://www.anaconda.com/download/>`_ Python package manager. Using ``conda``,
you can install GSL with::

    conda install -c conda-forge gsl


Installing with ``apt``
-----------------------

If you use Linux, you can install GSL with anaconda (see directions above), or
with ``apt``. To install with apt, make sure to install both ``gsl-bin`` and
``libgsl0-dev``::

    apt-get install gsl-bin libgsl0-dev


Forcing gala to install without GSL support
-------------------------------------------

You can force Gala to build without GSL support using the ``--nogsl`` flag
passed to setup.py. To use this flag, you must install Gala from source by
cloning the repository (see above) and running::

    python -m pip install gala --install-option="--nogsl"


Python Dependencies
===================

This packages has the following dependencies:

* `Python`_ >= 3.7
* `Numpy`_
* `Cython <http://www.cython.org/>`_
* `Astropy`_
* `PyYAML`_
* `scipy`_

Explicit version requirements are specified in the project `setup.cfg
<https://github.com/adrn/gala/blob/main/setup.cfg>`_. ``pip`` and ``conda``
should install and enforce these versions automatically.

Optional
--------

- `Sympy`_ for creating :class:`~gala.potential.potential.PotentialBase`
  subclass instances from a mathematical expression using
  :func:`~gala.potential.potential.from_equation()`.
- ``galpy``
- `h5py`
- `matplotlib`
-