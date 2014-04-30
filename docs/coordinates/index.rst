.. _astropy-coordinates:

*********************************************
Coordinate Systems (`streamteam.coordinates`)
*********************************************

Introduction
============

The `streamteam.coordinates` package provides utility functions for converting
coordinates and velocities between various astronomical systems.

.. warning::
    `streamteam.coordinates` is currently a work-in-progress, and thus it is
    possible there will be significant API changes in later versions.


Getting Started
===============

The utility functions in `streamteam.coordinates` make use of the coordinates
subpackage in astropy, `astropy.coordinates <http://docs.astropy.org/en/latest/coordinates/>`_.
So far, only a few functions are implemented:

- Converting a velocity from/to the Galactic Standard of Rest (GSR) to/from a
  heliocentric velocity.
- Converting a position (and velocity) from/to Galactic cartesian coordinates
  to/from Heliocentric spherical coordinates.

These functions work naturally with the `astropy <http://www.astropy.org>`_ unit
system and coordinate subpackage::

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import streamteam.coordinates as stc
    >>> c = coord.ICRS(ra=10.68458*u.degree, dec=41.26917*u.degree, distance=15*u.kpc)
    >>> stc.hel_to_gal_xyz(c)
    <Quantity [-15.22074498, 11.93494427, -5.5152468 ] kpc>
    >>> stc.vhel_to_vgsr(c, 110*u.km/u.s)
    <Quantity 281.7729618014674 km / s>

Also included are coordinate classes for transforming to Sagittarius and Orphan
stream coordinates, as defined in the references below. These classes behave like
the native astropy coordinates classes (e.g., `ICRS`) and can be transformed to
and from other astropy coordinate objects::

    >>> from streamteam.coordinates import Sagittarius
    >>> c = coord.ICRS(ra=10.68458*u.degree, dec=41.26917*u.degree)
    >>> c.transform_to(Sagittarius)
    <Sagittarius Lambda=113.81238 deg, Beta=-45.82266 deg>
    >>> s = Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> s.transform_to(coord.Galactic)
    <Galactic l=182.59221 deg, b=-9.53969 deg>

References
==========

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_

Reference/API
=============
.. autofunction:: streamteam.coordinates.vgsr_to_vhel
.. autofunction:: streamteam.coordinates.vhel_to_vgsr
.. autofunction:: streamteam.coordinates.gal_xyz_to_hel
.. autofunction:: streamteam.coordinates.hel_to_gal_xyz
