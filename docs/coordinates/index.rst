.. _astropy-coordinates:

*********************************************
Coordinate Systems (`streamteam.coordinates`)
*********************************************

Introduction
============

The `streamteam.coordinates` package provides functions for converting
coordinates and velocities between various astronomical systems, as well
as `astropy.coordinates` frame classes for coordinates defined by the
Sagittarius and Orphan streams.

.. warning::
    `streamteam.coordinates` is currently a work-in-progress, and thus it is
    possible there will be significant API changes in later versions.


Getting Started
===============

The functions in this subpackage make use of the coordinates subpackage in
astropy, `astropy.coordinates <http://docs.astropy.org/en/latest/coordinates/>`_.
Currently available are functions to:

- Convert a velocity from/to the Galactic Standard of Rest (GSR) to/from a
  heliocentric velocity.
- Convert a position (and velocity) from/to Galactic cartesian coordinates
  to/from Heliocentric spherical coordinates.

These functions work naturally with the `astropy <http://www.astropy.org>`_ unit
system and coordinate subpackages. For example, to convert a sky position
and distance to a Galactocentric, cartesian position, we first have to define
an astropy coordinate frame object::

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> c = coord.ICRS(ra=100.68458*u.degree, dec=41.26917*u.degree, distance=15*u.kpc)

Then pass this object in to the heliocentric to galactocentric conversion
function::

    >>> import streamteam.coordinates as stc
    >>> stc.hel_to_gal_xyz(c)
    <Quantity [-22.34899301,  1.42957337,  4.13070449] kpc>

Or, with the same sky position (as specified by the astropy frame object)
and a radial velocity, we could convert to the Galactic standard of rest::

    >>> stc.vhel_to_vgsr(c, 110*u.km/u.s)
    <Quantity 123.87590811841189 km / s>

Both of these functions allow specifying the circular velocity at the Sun
(`vcirc`) and a 3-vector specifying the Sun's velocity with respect to the
local standard of rest (`vlsr`). The position transformation functions also
allow specifying the distance of the Sun to the Galactic center (`xsun`)::

    >>> stc.hel_to_gal_xyz(c, xsun=-8.3*u.kpc)
    <Quantity [-22.64899301,  1.42957337,  4.13070449] kpc>

Also included are coordinate classes for transforming to Sagittarius and Orphan
stream coordinates (as defined in the references below). These classes behave
like the built-in astropy coordinates frames (e.g., `ICRS` or `Galactic`) and
can be transformed to and from other astropy coordinate frames::

    >>> from streamteam.coordinates import Sagittarius
    >>> c = coord.ICRS(ra=10.68458*u.degree, dec=41.26917*u.degree)
    >>> c.transform_to(Sagittarius)
    <Sagittarius Coordinate: (Lambda, Beta, distance) in (deg, deg, )
        (113.8123808711934, -45.822661836833824, 1.0)>
    >>> s = Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> s.transform_to(coord.Galactic)
    <Galactic Coordinate: (l, b, distance) in (deg, deg, )
        (182.5922090437946, -9.539692094685897, 1.0)>

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
.. autoclass:: streamteam.coordinates.Sagittarius
.. autoclass:: streamteam.coordinates.Orphan
