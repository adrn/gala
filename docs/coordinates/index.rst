.. _coordinates:

*********************************************
Coordinate Systems (`gary.coordinates`)
*********************************************

Introduction
============

The `gary.coordinates` package provides functions for converting
coordinates and velocities between various astronomical systems, as well
as `astropy.coordinates` frame classes for coordinates defined by the
Sagittarius and Orphan streams.

.. warning::
    `gary.coordinates` is currently a work-in-progress, and thus it is
    possible there will be significant API changes in later versions.


Getting Started
===============

The functions in this subpackage make use of the coordinates subpackage in
Astropy, `astropy.coordinates <http://docs.astropy.org/en/latest/coordinates/>`_.
Currently available are functions to:

- Convert a velocity from/to the Galactic Standard of Rest (GSR) to/from a
  heliocentric (LSR) velocity.
- Convert a position (and velocity) from/to Galactic cartesian coordinates
  to/from Heliocentric spherical coordinates.

These functions work naturally with the `Astropy <http://www.astropy.org>`_ unit
system and coordinate subpackages. For example, to convert a sky position
and distance to a Galactocentric, cartesian position, we first have to define
an Astropy coordinate::

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> c = coord.SkyCoord(ra=100.68458*u.degree, dec=41.26917*u.degree, distance=15*u.kpc)

Then pass this object in to the heliocentric to galactocentric conversion
function::

    >>> import gary.coordinates as stc
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

These functions also work on objects containing multiple coordinates::

    >>> c = coord.SkyCoord(ra=[100.68458, 41.23123]*u.degree,
    ...                    dec=[41.26917, -11.412]*u.degree,
    ...                    distance=[13.412, 1534.1231]*u.kpc)
    <Quantity [[ -2.11299130e+01, -7.89201791e+02],
               [  1.27822920e+00, -1.08828561e+02],
               [  3.69340057e+00, -1.31601004e+03]] kpc>

Also included are coordinate classes for transforming to Sagittarius and Orphan
stream coordinates (as defined in the references below). These classes behave
like the built-in astropy coordinates frames (e.g., `ICRS` or `Galactic`) and
can be transformed to and from other astropy coordinate frames::

    >>> c = coord.SkyCoord(ra=100.68458*u.degree, dec=41.26917*u.degree)
    >>> c.transform_to(stc.Sagittarius)
    <SkyCoord (Sagittarius): (Lambda, Beta, distance) in (deg, deg, )
        (179.5851618648458, -12.558369149811035, 1.0)>
    >>> s = stc.Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> c = coord.SkyCoord(s)
    >>> c.galactic
    <SkyCoord (Galactic): (l, b, distance) in (deg, deg, )
        (182.5922090437946, -9.539692094685897, 1.0)>

References
==========

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_

Reference/API
=============
.. autofunction:: gary.coordinates.vgsr_to_vhel
.. autofunction:: gary.coordinates.vhel_to_vgsr
.. autofunction:: gary.coordinates.gal_xyz_to_hel
.. autofunction:: gary.coordinates.hel_to_gal_xyz
.. autoclass:: gary.coordinates.Sagittarius
.. autoclass:: gary.coordinates.Orphan
