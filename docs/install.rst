.. include:: references.txt

.. _gala-install:

============
Installation
============

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

This packages has the following dependencies:

- `Python`_ >= 2.7

- `Numpy`_ >= 1.8

- `Cython <http://www.cython.org/>`_: >= 0.23

- `Astropy`_ >= 1.1

- `PyYAML`_ >= 3.10

You can use ``pip`` or ``conda`` to install these automatically.

Optional
--------

- `Sympy`_ for creating `~gala.potential.PotentialBase` objects from a
    mathematical expression using `~gala.potential.from_equation()`.
