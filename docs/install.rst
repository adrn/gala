************
Installation
************

Requirements
============

This packages has the following strict requirements:

- `Python <http://www.python.org/>`_ 2.7 (untested with versions >2.7.8)

- `Numpy <http://www.numpy.org/>`_ 1.7 or later

- `Cython <http://www.cython.org/>`_: 0.21 or later

- `Astropy <http://www.astropy.org/>`_ 1.0 or later

You can use pip to install these automatically using the [pip-requirements.txt](https://github.com/adrn/gary/blob/master/pip-requirements-txt) file (from the root of the project):

    pip install -r pip-requirements.txt

Optional
--------

For faster FFT's in the NAFF routines (:mod:`gary.dynamics`), install the
``FFTW`` library and the Python bindings, ``pyFFTW``.

Installing
==========

Development repository
----------------------

The latest development version of gary can be cloned from
[GitHub](https://github.com/) using git::

   git clone git://github.com/adrn/gary.git

Building and Installing
-----------------------

To build the project (from the root of the source tree, e.g., inside
the cloned ``gary`` directory)::

    python setup.py build

To install the project::

    python setup.py install

Building the documentation
--------------------------

This requires a few extra pip-installable packages, listed in the [docs-pip-requirements.txt](https://github.com/adrn/gary/blob/master/docs-pip-requirements-txt). You can install these extra dependencies automatically using pip (from the root of the project):

    pip install -r docs-pip-requirements.txt
