.. include:: references.txt

.. _gala-install:

============
Installation
============

With ``conda``
==============

To install the latest stable version with ``conda``, use the ``conda-forge``
channel with::

    conda install -c conda-forge astro-gala

With ``pip``
============

To install the latest stable version using ``pip``, use::

    pip install astro-gala

To install the development version::

    pip install git+https://github.com/adrn/gala

Cloning, Building, Installing
=============================

The latest development version of gala can be cloned from
`GitHub <https://github.com/>`_ using ``git``::

   git clone git://github.com/adrn/gala.git

To build the project (from the root of the source tree, e.g., inside
the cloned ``gala`` directory)::

    python setup.py build

To install the project::

    python setup.py install


GSL support
===========

Some functionality in Gala depends on the GNU Scientific Library (GSL), a C
library for numerical and mathematical programming. By default, Gala will
determine whether to install with or without GSL support depending on whether it
can find a GSL installation on your machine. If you are not sure whether you
have GSL installed or not, try running:

    gsl-config --version

in your terminal. If that returns a version number, you likely have GSL
installed. If it errors, you will need to install it. Additionally, if your
version of GSL is <1.14, we recommend updating to a newer version, as Gala has
only been tested with GSL >= 1.14.

Installing with ``conda``
-------------------------

If you use a Mac computer, we recommend installing GSL using the `anaconda
<https://www.anaconda.com/download/>`_ Python package manager. Using ``conda``,
you can install GSL with:

    conda install -c conda-forge gsl


Installing with ``apt``
-----------------------

If you use Linux, you can install GSL with anaconda (see directions above), or
with ``apt``. To install with apt, make sure to install both ``gsl-bin`` and
``libgsl0-dev``:

    apt-get install gsl-bin libgsl0-dev


Forcing gala to install without GSL support
-------------------------------------------

You can force Gala to build without GSL support using the ``--nogsl`` flag
passed to setup.py. To use this flag, you must install Gala from source by
cloning the repository (see above) and running:

    python setup.py build --nogsl
    python setup.py install


Python Dependencies
===================

This packages has the following dependencies (note that the version requirements
below indicate the versions for which Gala is tested with and should work with):

- `Python`_ >= 3.6

- `Numpy`_ >= 1.16

- `Cython <http://www.cython.org/>`_: >= 0.28

- `Astropy`_ >= 3

- `PyYAML`_ >= 3.10

- `scipy`_ >= 1.1

You can use ``pip`` or ``conda`` to install these automatically.

Optional
--------

- `Sympy`_ for creating `~gala.potential.PotentialBase` objects from a
    mathematical expression using `~gala.potential.from_equation()`.
