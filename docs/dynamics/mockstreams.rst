.. _mockstreams:

*******************************
Generating mock stellar streams
*******************************

Introduction
============

This module contains functions for generating mock stellar streams using a variety
of approximate methods. TODO: e.g., streakline, particle-spray, etc...

Some imports needed for the code below::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gary.potential as gp
    >>> import gary.dynamics as gd
    >>> import gary.dynamics as gd
    >>> from gary.units import galactic

TODO
============================

Fardal et al. (2015) parametrization
------------------------------------

.. plot::
    :align: center



Streakline method
-----------------


API
---
.. automodapi:: gary.dynamics.mockstream
    :no-heading:
    :headings: ^^
