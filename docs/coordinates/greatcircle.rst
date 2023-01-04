For the examples below, we assume the following imports have already been
executed::

    >>> import astropy.units as u
    >>> import astropy.coordinates as coord
    >>> import numpy as np
    >>> import gala.coordinates as gc

.. _greatcircle:

*************************************************
Great circle and stellar stream coordinate frames
*************************************************


Introduction
============

Great circle coordinate systems are defined as a rotation from another spherical
coordinate system, such as the ICRS. The great circle system is defined by a specified
(north) pole and spherical origin -- i.e. a specification of the new coordinate system x
and z axes in components of the old coordinate system.

`gala` currently supports great circle frames that are defined as a rotation away from
the ICRS (RA, Dec) through the `~gala.coordinates.GreatCircleICRSFrame` class. To create
a new great circle frame with the default initializer, you must specify a pole using the
``pole`` keyword argument and the spherical origin with the ``origin`` argument.
However, this frame also supports other initialization paths through the ``from_``
classmethods (see API below). These classmethods are the most useful initialization
methods. For example, to define a great circle system with the pole at (RA, Dec) =
(32.5, 19.8)º and a longitude 0 at RA=100º, we first have to create a coordinate object
for the pole::

    >>> pole = coord.SkyCoord(ra=32.5*u.deg, dec=19.8*u.deg)

We can then pass this pole to the `~gala.coordinates.GreatCircleICRSFrame.from_pole_ra0`
classmethod to define our coordinate frame::

    >>> frame = gc.GreatCircleICRSFrame.from_pole_ra0(pole=pole, ra0=100*u.deg)

This frame instance acts like any other Astropy coordinate frame. For example, we can
transform other coordinates to this new coordinate system using::

    >>> c = coord.SkyCoord(ra=[160, 53]*u.deg, dec=[-11, 9]*u.deg)
    >>> c_fr = c.transform_to(frame)
    >>> c_fr # doctest: +FLOAT_CMP
    <SkyCoord (GreatCircleICRSFrame: pole=<ICRS Coordinate: (ra, dec) in deg
        (32.5, 19.8)>, origin=<ICRS Coordinate: (ra, dec) in deg
        (280., 46.74765478)>, priority=origin): (phi1, phi2) in deg
        [(-127.59199268, -38.82050866), (-154.93887946,  67.43382209)]>

The spherical coordinate components of the resulting great circle frame are
always named ``phi1`` and ``phi2``, so to access the longitude and latitude in
the new system, we use::

    >>> c_fr.phi1 # doctest: +FLOAT_CMP
    <Longitude [-127.59199268, -154.93887946] deg>
    >>> c_fr.phi2 # doctest: +FLOAT_CMP
    <Latitude [-38.82050866,  67.43382209] deg>

The transformation also works for velocity components. For example, if we have a
sky position and proper motions, we can transform to the great circle frame in
the same way::

    >>> c2 = coord.SkyCoord(
    ...     ra=160*u.deg,
    ...     dec=-11*u.deg,
    ...     pm_ra_cosdec=5*u.mas/u.yr,
    ...     pm_dec=0.3*u.mas/u.yr
    ... )
    >>> c2_fr = c2.transform_to(frame)
    >>> c2_fr.phi1 # doctest: +FLOAT_CMP
    <Longitude -127.59199268 deg>
    >>> c2_fr.pm_phi1_cosphi2 # doctest: +FLOAT_CMP
    <Quantity 1.71997614 mas / yr>
    >>> c2_fr.pm_phi2 # doctest: +FLOAT_CMP
    <Quantity -4.70443217 mas / yr>

The generic great circle frame can also handle transforming from great circle
coordinates to other coordinate frames. For example, to transform a grid of points along
a great circle to the ICRS system, we would define a frame with positional data and a
specified pole::

    >>> c3_fr = gc.GreatCircleICRSFrame(
    ...     phi1=np.linspace(0, 360, 8)*u.deg,
    ...     phi2=0*u.deg,
    ...     pole=frame.pole,
    ...     origin=frame.origin
    ... )
    >>> c3 = c3_fr.transform_to(coord.ICRS())
    >>> c3.ra # doctest: +FLOAT_CMP
    <Longitude [280.        , 302.73861084, 326.04009238,  67.95460569,
                113.51995793, 132.05271289, 180.05477998, 280.        ] deg>


Creating a coordinate frame from two points along a great circle
================================================================

It is sometimes convenient to define a great circle coordinate frame by specifying two
endpoints of an arc segment along a great circle (instead of the pole). For these use
cases, the `~gala.coordinates.GreatCircleICRSFrame.from_endpoints` provides a
convenience classmethod for creating a great circle frame with endpoints::

    >>> endpoints = coord.SkyCoord(
    ...     ra=[-38.8, 4.7]*u.deg,
    ...     dec=[-45.1, -51.7]*u.deg
    ... )
    >>> frame2 = gc.GreatCircleICRSFrame.from_endpoints(endpoints[0], endpoints[1])
    >>> frame2
    <GreatCircleICRSFrame Frame (pole=<ICRS Coordinate: (ra, dec) in deg
        (359.1291976, 38.16814051)>, origin=<ICRS Coordinate: (ra, dec) in deg
        (341.46580563, -50.48035324)>, priority=origin)>

Without specifying a longitude zeropoint, the default behavior of the above classmethod
is to take the spherical midpoint of the two endpoints as the longitude zeropoint.
However, a custom zeropoint can be specified using the ``ra0`` keyword argument. For
example::

    >>> frame3 = gc.GreatCircleICRSFrame.from_endpoints(
    ...     endpoints[0], endpoints[1], ra0=150*u.deg
    ... )
    >>> frame3
    <GreatCircleICRSFrame Frame (pole=<ICRS Coordinate: (ra, dec) in deg
        (359.1291976, 38.16814051)>, origin=<ICRS Coordinate: (ra, dec) in deg
        (330., -48.01820335)>, priority=origin)>


Creating a coordinate frame from endpoints and an origin
========================================================

When working with stellar streams, it is sometimes useful to create a stream-aligned
coordinate frame by specifying an exact origin for the new great circle coordinate frame
(e.g., set to the progenitor system) along with the endpoints of the stream (which are
often close to defining a great circle). In these cases, the great circle defined by the
endpoints and the great circle defined by the origin may not be orthogonal. You can
still use these to create a `~gala.coordinates.GreatCircleICRSFrame`, but by default the
pole location will be adjusted to be orthogonal to the input origin::

    >>> endpoints = coord.SkyCoord(
    ...     ra=[-38.8, 4.7]*u.deg,
    ...     dec=[-45.1, -51.7]*u.deg
    ... )
    >>> origin = coord.SkyCoord(330., -48., unit=u.deg)
    >>> frame4 = gc.GreatCircleICRSFrame.from_endpoints(  # doctest: +IGNORE_WARNINGS
    ...     endpoints[0], endpoints[1], origin=origin
    ... )
    >>> frame4
    <GreatCircleICRSFrame Frame (pole=<ICRS Coordinate: (ra, dec) in deg
        (359.13616655, 38.18404071)>, origin=<ICRS Coordinate: (ra, dec) in deg
        (330., -48.)>, priority=origin)>


Creating a coordinate frame from a pole and longitude zero point
================================================================

Another common way of initializing great circle coordinate systems is with a pole and a
longitude zero point (as was previously — prior to v1.7 — allowed in the initializer
`~gala.coordinates.GreatCircleICRSFrame`). This can now be done with the
`~gala.coordinates.GreatCircleICRSFrame.from_pole_ra0` classmethod::

    >>> frame5 = gc.GreatCircleICRSFrame.from_pole_ra0(
    ...     pole=pole, ra0=100*u.deg
    ... )
    >>> frame5
    <GreatCircleICRSFrame Frame (pole=<ICRS Coordinate: (ra, dec) in deg
        (32.5, 19.8)>, origin=<ICRS Coordinate: (ra, dec) in deg
        (280., 46.74765478)>, priority=origin)>

With just these inputs, there is an ambiguity in the definition of the coordinate frame
because the great circles defined by the pole and longitude zero point intersect at two
locations (so there are two possible origins, one being the negative of the other). The
convention here is to pick the origin closest to (0, 0). To have finer control over
which origin is picked, you can also pass in a sky coordinate object with the
``origin_disambiguate`` argument, and the origin closest to this coordinate will be used
to define the coordinate frame.


.. _greatcircle-api:

API
===

.. automodapi:: gala.coordinates.greatcircle
    :no-inheritance-diagram:
