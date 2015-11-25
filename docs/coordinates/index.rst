.. include:: references.txt

.. _gary-coordinates:

*********************************************
Coordinate Systems (`gary.coordinates`)
*********************************************

Introduction
============

The ``gary.coordinates`` subpackage provides functions for converting velocities between
various astronomical coordinate frames and systems. This subpackage also provides
:mod:`astropy.coordinates` frame classes for coordinate sytems defined by the Sagittarius and
Orphan streams.

.. warning::
    `gary.coordinates` is currently a work-in-progress, and thus it is
    possible there will be API changes in later versions.

Getting Started
===============

The core functions in this subpackage provide support to:

- Convert Galactocentric, cartesian velocities to heliocentric proper motion
  and radial velocities.
- Convert proper motions and radial velocities to Galactocentric, cartesian
  velocities.
- Convert proper motions from/to ICRS to/from Galactic.
- Convert radial velocities from/to the Galactic Standard of Rest (GSR) to/from a
  barycentric frame.

These functions work naturally with the :mod:`astropy.units` and
:mod:`astropy.coordinates` subpackages. Handling positional transformations is already
supported by :mod:`astropy.coordinates` and new to Astropy v1.0 is a
:class:`~astropy.coordinates.Galactocentric` reference frame. However, there is currently
no support in Astropy for transforming velocities. The functions below attempt to bridge
that gap as a temporary solution until support is added (planned for v1.1).

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
    <Quantity [-134.44094022, 228.42957796,  52.97041271] km / s>

Because the input coordinate is given in the ICRS frame, the function assumes that
the proper motion is also in this frame, e.g., that the proper motion components are
:math:`(\mu_\alpha\cos\delta, \mu_\delta)`. If we instead passed in a coordinate in
the Galactic frame, the components are assumed to be :math:`(\mu_l\cos b, \mu_b)`::

    >>> gc.vhel_to_gal(c.galactic, pm=pm, rv=rv)
    <Quantity [-137.63839042, 232.10966635,  40.73819003] km / s>

The velocity transformation functions allow specifying the circular velocity at the Sun
(`vcirc`) and a 3-vector specifying the Sun's velocity with respect to the local
standard of rest (`vlsr`). Further customization of the Sun's location can be made via
the :class:`~astropy.coordinates.Galactocentric` frame attributes and passed in with the
keyword argument ``galactocentric_frame``::

    >>> frame = coord.Galactocentric(z_sun=10.*u.pc, galcen_distance=8.3*u.kpc)
    >>> gc.vhel_to_gal(c.icrs, pm=pm, rv=rv, galactocentric_frame=frame,
    ...                vcirc=218*u.km/u.s, vlsr=[0.,0.,0.]*u.km/u.s)
    <Quantity [-144.5344455 , 221.17957796,  45.50447318] km / s>

The inverse transformations are also available, with the function
:func:`~gary.coordinates.vgal_to_hel`. Here, because the input coordinate is passed
in after being transformed to the ICRS frame, the output proper motions will also be
given in the ICRS frame :math:`(\mu_\alpha\cos\delta, \mu_\delta)`::

    >>> xyz = coord.Galactocentric([11., 15, 25] * u.kpc)
    >>> vxyz = [121., 150., -75.] * u.km/u.s
    >>> gc.vgal_to_hel(xyz.transform_to(coord.ICRS), vxyz)
    (<Quantity 0.2834641666390529 mas / yr>, <Quantity -0.888174413651107 mas / yr>, <Quantity -29.71790624810498 km / s>)

Passing in coordinates in the Galactic frame means that the output proper motions will
instead be :math:`(\mu_l\cos b, \mu_b)`::

    >>> gc.vgal_to_hel(xyz.transform_to(coord.Galactic), vxyz)
    (<Quantity -0.7713637315333076 mas / yr>, <Quantity -0.5236445726220675 mas / yr>, <Quantity -29.717906248104974 km / s>)

All of these functions also work on arrays of coordinates and velocities, e.g.::

    >>> import numpy as np
    >>> xyz = coord.Galactocentric(np.random.uniform(-20,20,size=(3,10)) * u.kpc)
    >>> vxyz = np.random.uniform(-150,150,size=(3,10)) * u.km/u.s
    >>> gc.vgal_to_hel(xyz.transform_to(coord.ICRS), vxyz) # doctest: +SKIP
    ...

Proper motion transformations
-----------------------------

Transforming between ICRS and Galactic proper motions is supported in Gary. To
demonstrate, we again need to first define a coordinate for the object of
interest::

    >>> c = coord.SkyCoord(ra=100.68458*u.deg, dec=41.26917*u.deg)

Now we define the proper motion as a :class:`~astropy.units.Quantity`, and pass
in to the relevant transformation function. Here, we will transform from ICRS
to Galactic::

    >>> pm = [1.53, -2.1] * u.mas/u.yr
    >>> gc.pm_icrs_to_gal(c, pm)
    <Quantity [ 2.52366087, 0.61809041] mas / yr>

Of course, these functions also work on arrays. The first axis of the input
proper motion arrays should have length=2::

    >>> ra = np.random.uniform(0,360,size=10) * u.degree
    >>> dec = np.random.uniform(-90,90,size=10) * u.degree
    >>> c = coord.SkyCoord(ra=ra, dec=dec)
    >>> pm = np.random.uniform(-10,10,size=(2,10)) * u.mas/u.yr
    >>> gc.pm_icrs_to_gal(c, pm) # doctest: +SKIP
    ...

Tidal Stream Coordinate Frames
------------------------------

Also included in this subpackage are Astropy coordinate frame classes for
transforming to Sagittarius and Orphan stream coordinates (as defined in the
references below). These classes behave like the built-in astropy coordinates
frames (e.g., :class:`~astropy.coordinates.ICRS` or
:class:`~astropy.coordinates.Galactic`) and can be transformed to and from
other astropy coordinate frames::

    >>> c = coord.SkyCoord(ra=100.68458*u.degree, dec=41.26917*u.degree)
    >>> c.transform_to(gc.Sagittarius)
    <SkyCoord (Sagittarius): (Lambda, Beta, distance) in (deg, deg, )
        (179.58511053544734, -12.558450192162631, 1.0)>
    >>> s = gc.Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> c = coord.SkyCoord(s)
    >>> c.galactic
    <SkyCoord (Galactic): (l, b, distance) in (deg, deg, )
        (182.5922090437946, -9.539692094685897, 1.0)>

References
==========

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_

.. _gary-coordinates-api:

API
===

.. automodapi:: gary.coordinates

