.. _coordinates:

*********************************************
Coordinate Systems (`gary.coordinates`)
*********************************************

Introduction
============

The ``gary.coordinates`` package provides functions for converting
coordinates and velocities between various astronomical systems, as well
as :mod:`astropy.coordinates` frame classes for coordinates defined by the
Sagittarius and Orphan streams.

.. warning::
    `gary.coordinates` is currently a work-in-progress, and thus it is
    possible there will be significant API changes in later versions.


Getting Started
===============

The functions in this subpackage make use of the coordinates subpackage in
Astropy, :mod:`astropy.coordinates`. Currently available are functions to:

- Convert Galactocentric, cartesian velocities to heliocentric proper motion
  and radial velocities.
- Convert proper motions and radial velocities to Galactocentric, cartesian
  velocities.
- Convert proper motions from/to ICRS to/from Galactic.
- Convert radial velocities from/to the Galactic Standard of Rest (GSR) to/from a
  barycentric frame.

These functions work naturally with the :mod:`astropy.units` and
:mod:`astropy.coordinates` subpackages. Handling positional transformations is already
supported by :mod:`astropy.coordinates`, but new to v1.0 is a
:class:`~astropy.coordinates.Galactocentric` reference frame. However, there is currently
no support in Astropy for transforming velocities. The functions below attempt to bridge that gap as a temporary solution until support is added (planned for v1.1).

For example, to convert a spherical, heliocentric velocity (proper motion and radial
velocity) in an ICRS frame to a Galactocentric, cartesian velocity, we first have
to define an Astropy coordinate for the position of the object::

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> c = coord.SkyCoord(ra=100.68458*u.deg, dec=41.26917*u.deg, distance=1.1*u.kpc)

Then pass this object in to the heliocentric to galactocentric conversion
function, :func:`~gary.coordinates.vhel_to_gal`::

    >>> import gary.coordinates as gc
    >>> pm = [1.5, -1.7] * u.mas/u.yr
    >>> rv = 151.1 * u.km/u.s
    >>> gc.vhel_to_gal(c.icrs, pm=pm, rv=rv)
    <Quantity [-22.34899301,  1.42957337,  4.13070449] kpc>

Because the input coordinate is given in the ICRS frame, the function assumes that
the proper motion is also in this frame, e.g., that the proper motion components are
:math:`(\mu_\alpha\cos\delta, \mu_\delta)`. If we instead passed in a coordinate in
the Galactic frame, the components are assumed to be :math:`(\mu_l\cos b, \mu_b)`::

    >>> gc.vhel_to_gal(c.galactic, pm=pm, rv=rv)
    <Quantity [-137.63839042,  232.10966635,  40.73819003] km / s>

The velocity transformation functions allow specifying the circular velocity at the Sun
(`vcirc`) and a 3-vector specifying the Sun's velocity with respect to the local
standard of rest (`vlsr`). Further customization of the Sun's location can be made via
the :class:`~astropy.coordinates.Galactocentric` frame attributes and passed in with the
keyword argument ``galactocentric_frame``::

    >>> frame = coord.Galactocentric(z_sun=10.*u.pc, galcen_distance=8.3*u.kpc)
    >>> gc.vhel_to_gal(c.icrs, pm=pm, rv=rv, galactocentric_frame=frame,
    ...                vcirc=218*u.km/u.s, vlsr=[0.,0.,0.]*u.km/u.s)
    <Quantity [-144.5344455 , 221.17957796,  45.50447318] km / s>

These functions also work on arrays of coordinates and velocities, e.g.::

    >>> c = coord.SkyCoord(ra=[100.68458, 41.23123]*u.degree,
    ...                    dec=[41.26917, -11.412]*u.degree,
    ...                    distance=[13.412, 1534.1231]*u.kpc)
    >>> TODO:
    <Quantity [[ -2.11299130e+01, -7.89201791e+02],
               [  1.27822920e+00, -1.08828561e+02],
               [  3.69340057e+00, -1.31601004e+03]] kpc>


Tidal Stream Coordinates
------------------------

Also included are coordinate classes for transforming to Sagittarius and Orphan
stream coordinates (as defined in the references below). These classes behave
like the built-in astropy coordinates frames (e.g.,
:class:`~astropy.coordinates.ICRS` or :class:`~astropy.coordinates.Galactic`) and
can be transformed to and from other astropy coordinate frames::

    >>> c = coord.SkyCoord(ra=100.68458*u.degree, dec=41.26917*u.degree)
    >>> c.transform_to(gc.Sagittarius)
    <SkyCoord (Sagittarius): (Lambda, Beta, distance) in (deg, deg, )
        (179.5851618648458, -12.558369149811035, 1.0)>
    >>> s = gc.Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> c = coord.SkyCoord(s)
    >>> c.galactic
    <SkyCoord (Galactic): (l, b, distance) in (deg, deg, )
        (182.5922090437946, -9.539692094685897, 1.0)>

References
==========

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_

Coordinate conversion
=====================

.. autosummary::
   :nosignatures:
   :toctree: _coordinates/

   gary.coordinates.vgal_to_hel
   gary.coordinates.vhel_to_gal
   gary.coordinates.pm_gal_to_icrs
   gary.coordinates.pm_icrs_to_gal
   gary.coordinates.vgsr_to_vhel
   gary.coordinates.vhel_to_vgsr

New coordinate classes
======================

.. autosummary::
   :toctree: _coordinates/
   :template: class.rst

   gary.coordinates.Sagittarius
   gary.coordinates.Orphan

