************
Installation
************

Requirements
============

This packages has the following strict requirements:

- `Python <http://www.python.org/>`_ 2.7 (untested with >3.0!)

- `Numpy <http://www.numpy.org/>`_ 1.7 or later

- `Cython <http://www.cython.org/>`_: 0.20 or later

- `Astropy <http://www.astropy.org/>`_ 0.4 or later

Installing
==========

Obtaining the source packages
-----------------------------

Development repository
^^^^^^^^^^^^^^^^^^^^^^

The latest development version of gary can be cloned from github
using this command::

   git clone git://github.com/adrn/gary.git

Building and Installing
-----------------------

To build the project (from the root of the source tree, e.g., inside
the cloned gary directory)::

    python setup.py build

To install the project::

    python setup.py install

Building the documentation
--------------------------

Requires a few extra pip-installable packages:

- `numpydoc`

- `sphinx-rtd-theme`
