.. include:: references.txt

.. _gala-install:

============
Installation
============

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

Dependencies
============

This packages has the following dependencies (note that the version requirements
below indicate the versions for which Gala is tested with and should work with):

- `Python`_ >= 2.7

- `Numpy`_ >= 1.12

- `Cython <http://www.cython.org/>`_: >= 0.25

- `Astropy`_ >= 2

- `PyYAML`_ >= 3.10

You can use ``pip`` or ``conda`` to install these automatically.

Optional
--------

- `Sympy`_ for creating `~gala.potential.PotentialBase` objects from a
    mathematical expression using `~gala.potential.from_equation()`.
