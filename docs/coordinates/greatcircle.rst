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
coordinate system, such as the ICRS. The great circle system is defined by a
specified north pole, with a additional (optional) specification of the
longitude zero point of the final system.

`gala` currentlt supports great circle frames that are defined as a rotation
away from the ICRS (ra, dec) through the
`~gala.coordinates.GreatCircleICRSFrame` class. To create a new great circle
frame, you must specify a pole using the ``pole=`` keyword, and optionally
specify the longitude zero point either by specifying the right ascension of the
longitude zero point, ``ra0``, or by specifying a final rotation to be applied
to the transformation, ``rotation``. For example, to define a great circle
system with the pole at (RA, Dec) = (32.5, 19.8)º, we first have to create a
coordinate object for the pole::

    >>> pole = coord.SkyCoord(ra=32.5*u.deg, dec=19.8*u.deg)

We then pass this pole to the `~gala.coordinates.GreatCircleICRSFrame` class to
define our coordinate frame::

    >>> fr = gc.GreatCircleICRSFrame(pole=pole)

We can then use this frame like any other Astropy coordinate frame. For example,
we can transform other coordinates to this new coordinate system using::

    >>> c = coord.SkyCoord(ra=[160, 53]*u.deg, dec=[-11, 9]*u.deg)
    >>> c_fr = c.transform_to(fr)
    >>> c_fr # doctest: +FLOAT_CMP
    <SkyCoord (GreatCircleICRSFrame: pole=<ICRS Coordinate: (ra, dec) in deg
        ( 32.5,  19.8)>, center=None, ra0=nan deg, rotation=0.0 deg): (phi1, phi2) in deg
        [(91.68381582, -38.82050866), (64.33692905,  67.43382209)]>

The spherical coordinate components of the resulting great circle frame are
always named ``phi1`` and ``phi2``, so to access the longitude and latitude in
the new system, we use::

    >>> c_fr.phi1 # doctest: +FLOAT_CMP
    <Longitude [91.68381582, 64.33692905] deg>
    >>> c_fr.phi2 # doctest: +FLOAT_CMP
    <Latitude [-38.82050866,  67.43382209] deg>

The transformation also works for velocity components. For example, if we have a
sky position and proper motions, we can transform to the great circle frame in
the same way::

    >>> c = coord.SkyCoord(ra=160*u.deg,
    ...                    dec=-11*u.deg,
    ...                    pm_ra_cosdec=5*u.mas/u.yr,
    ...                    pm_dec=0.3*u.mas/u.yr)
    >>> c_fr = c.transform_to(fr)
    >>> c_fr.phi1 # doctest: +FLOAT_CMP
    <Longitude 91.68381582 deg>
    >>> c_fr.pm_phi1_cosphi2 # doctest: +FLOAT_CMP
    <Quantity 1.71997614 mas / yr>
    >>> c_fr.pm_phi2 # doctest: +FLOAT_CMP
    <Quantity -4.70443217 mas / yr>

The generic great circle frame can also handle transforming from great circle
coordinates to other coordinate frames. For example, to transform a grid of
points along a great circle to the ICRS system, we would define a frame with
positional data and a specified pole::

    >>> c_fr = gc.GreatCircleICRSFrame(phi1=np.linspace(0, 360, 8)*u.deg,
    ...                                phi2=0*u.deg,
    ...                                pole=pole)
    >>> c = c_fr.transform_to(coord.ICRS)
    >>> c.ra # doctest: +FLOAT_CMP
    <Longitude [ 32.5       , 107.38324367, 126.92101217, 157.62242086,
                267.37757914, 298.07898783, 317.61675633,  32.5       ] deg>


Creating a great circle frame from two points
==============================================

It is sometimes convenient to specify two endpoints that define a great circle
instead of the pole. For such use cases, the
`~gala.coordinates.GreatCircleICRSFrame` has a convenience method for creating a
class from two endpoints of an arc that define a great circle::

    >>> points = coord.SkyCoord(ra=[-38.8, 4.7]*u.deg,
    ...                         dec=[-45.1, -51.7]*u.deg)
    >>> fr = gc.GreatCircleICRSFrame.from_endpoints(points[0], points[1])

Without specifying a longitude zeropoint, the default behavior of the above
method is to take the spherical midpoint of the two points as the zeropoint.
However, a custom zeropoint can be specified using the ``ra0`` or ``rotation``
keyword arguments. For example::

    >>> fr = gc.GreatCircleICRSFrame.from_endpoints(points[0], points[1],
    ...                                             ra0=150*u.deg)

.. _greatcircle-api:

API
===

.. automodapi:: gala.coordinates.greatcircle
    :no-inheritance-diagram:
