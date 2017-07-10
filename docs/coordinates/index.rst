.. _coordinates:

*********************************************
Coordinate Systems (`gala.coordinates`)
*********************************************

Introduction
============

The `~gala.coordinates` subpackage primarily provides specialty
:mod:`astropy.coordinates` frame classes for coordinate systems defined by the
Sagittarius, Orphan, and GD1 streams. It also contains functions for converting
velocities between various astronomical coordinate frames and systems, but these
are now deprecated with v0.2 because Astropy v2.0 `now supports velocity
transformations
<http://docs.astropy.org/en/stable/coordinates/velocities.html>`_.

For the examples below the following imports have already been executed::

    >>> import numpy as np
    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import gala.coordinates as gc

Stellar stream coordinate frames
================================

`Gala` provides Astropy coordinate frame classes for
transforming to Sagittarius, Orphan, and GD1 stream coordinates (as defined in the
references below). These classes behave like the built-in astropy coordinates
frames (e.g., :class:`~astropy.coordinates.ICRS` or
:class:`~astropy.coordinates.Galactic`) and can be transformed to and from
other astropy coordinate frames. For example, to convert a set of
`~astropy.coordinates.ICRS` (RA, Dec) coordinates to the
`~gala.coordinates.Sagittarius` frame::

    >>> c = coord.ICRS(ra=100.68458*u.degree, dec=41.26917*u.degree)
    >>> sgr = c.transform_to(gc.Sagittarius)
    >>> (sgr.Lambda, sgr.Beta) # doctest: +FLOAT_CMP
    (<Longitude 179.58511053544734 deg>, <Latitude -12.558450192162654 deg>)

Or, to transform from `~gala.coordinates.Sagittarius` coordinates to the
`~astropy.coordinates.Galactic` frame::

    >>> sgr = gc.Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> c = sgr.transform_to(coord.Galactic)
    >>> (c.l, c.b) # doctest: +FLOAT_CMP
    (<Longitude 182.5922090437946 deg>, <Latitude -9.539692094685893 deg>)

These transformations also handle velocities so that proper motion components
can be transformed between the systems. For example, to transform from
`~gala.coordinates.GD1` proper motions to `~astropy.coordinates.Galactic` proper
motions::

    >>> gd1 = gc.GD1(phi1=-35.00*u.degree, phi2=0*u.degree,
    ...              pm_phi1_cosphi2=-12.20*u.mas/u.yr,
    ...              pm_phi2=-3.10*u.mas/u.yr)
    >>> gd1.transform_to(coord.Galactic) # doctest: +FLOAT_CMP
    <Galactic Coordinate: (l, b) in deg
        ( 181.28968151,  54.84972806)
     (pm_l_cosb, pm_b) in mas / yr
        ( 12.03209393, -3.69847479)>

As with the other Astropy coordinate frames, with a full specification of the 3D
position and velocity, we can transform to a
`~astropy.coordinates.Galactocentric` frame::

    >>> gd1 = gc.GD1(phi1=-35.00*u.degree, phi2=0.04*u.degree,
    ...              distance=7.83*u.kpc,
    ...              pm_phi1_cosphi2=-12.20*u.mas/u.yr,
    ...              pm_phi2=-3.10*u.mas/u.yr,
    ...              radial_velocity=-32*u.km/u.s)
    >>> gd1.transform_to(coord.Galactocentric) # doctest: +FLOAT_CMP
    <Galactocentric Coordinate (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
        ( 266.4051, -28.936175)>, galcen_distance=8.3 kpc, galcen_v_sun=( 11.1,  232.24,  7.25) km / s, z_sun=27.0 pc, roll=0.0 deg): (x, y, z) in kpc
        (-12.78977138, -0.09870921,  6.44110283)
     (v_x, v_y, v_z) in km / s
        (-73.01933674, -216.37648654, -97.60065189)>

References
----------

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the
  Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_
* `Constraining the Milky Way potential with a 6-D phase-space map of the GD-1
  stellar stream <https://arxiv.org/abs/0907.1085>`_

Moving from ``gala.coordinates`` to ``astropy.coordinates``
===========================================================

``Gala`` previously supported the following transformations:

- Convert Galactocentric, cartesian velocities to heliocentric proper motion
  and radial velocities.
- Convert proper motions and radial velocities to Galactocentric, cartesian
  velocities.
- Convert proper motions from/to ICRS to/from Galactic.
- Convert radial velocities from/to the Galactic Standard of Rest (GSR) to/from
  a barycentric frame.

Below, we'll show examples of how to do each of these transformations using
``astropy.coordinates`` instead.

Convert Galactocentric (or simulated) to Heliocentric coordinates and velocities
--------------------------------------------------------------------------------

Let's assume we have a set of Cartesian positions and velocities that we
assume are in a Galactocentric (i.e. Milky Way-centric) frame::

    >>> xyz = [[  6.47945349, -34.4772621 ],
    ...        [-17.77019357,  31.11681441],
    ...        [-25.32101661,   3.54273331]] * u.kpc
    >>> vxyz = [[-184.32952533,   76.94467146],
    ...         [ -19.19375771,  -98.15603007],
    ...         [  22.49696323,   26.35110153]] * u.km/u.s

These could be, for example, the output from a simulation or sampling from a
distribution function. We want to transform these coordinates to a Helio- or
Barycentric coordinate frame to compute observable quantities like proper
motions and radial velocities. This is now supported in Astropy v2.0. To start,
we have to define the Galactocentric frame. That is, we have to define the sun's
position and velocity within the assumed Galactocentric frame. We do this by
setting frame attributes of the `astropy.coordinates.Galactocentric` frame. This
is a right-handed coordinate system with defaults for the position of the
Galactic center in ICRS coordinates, the sun-galactic-center distance, the
height of the sun above the Galactic midplane, and the solar velocity vector. We
can modify all of these parameters, but for the sake of example we'll just
change the distance and solar velocity (full velocity of the sun, including the
circular velocity and motion with respect to the local standard of rest)::

    >>> v_sun = coord.CartesianDifferential([10, 250., 7] * u.km/u.s)
    >>> gc = coord.Galactocentric(x=xyz[0], y=xyz[1], z=xyz[2],
    ...                           v_x=vxyz[0], v_y=vxyz[1], v_z=vxyz[2],
    ...                           galcen_distance=8*u.kpc,
    ...                           galcen_v_sun=v_sun)

To transform to any other `astropy.coordinates` frame, we use the
``transform_to()`` method::

    >>> icrs = gc.transform_to(coord.ICRS)

From the new frame, we can access the sky positions, Barycentric distances,
proper motion components, and Barycentric radial velocity. For example::

    >>> icrs.distance # doctest: +FLOAT_CMP
    <Distance [ 34.17535632, 41.00810092] kpc>
    >>> icrs.pm_ra_cosdec # doctest: +FLOAT_CMP
    <Quantity [ 1.78185606, 0.92782761] mas / yr>
    >>> icrs.radial_velocity # doctest: +FLOAT_CMP
    <Quantity [  46.14528132,-305.74417355] km / s>

Convert Heliocentric (observed) to Galactocentric coordinates and velocities
----------------------------------------------------------------------------

To transform from Heliocentric coordinates to Galactocentric coordinates, we
also have to define the sun's position and velocity within the assumed
Galactocentric frame. For the example below, we'll use the same frame parameter
values as above, but note that here we don't pass data in to the frame class::

    >>> v_sun = coord.CartesianDifferential([10, 250., 7] * u.km/u.s)
    >>> gc = coord.Galactocentric(galcen_distance=8*u.kpc,
    ...                           galcen_v_sun=v_sun)

Now we need some Heliocentric coordinates to transform. The Galactocentric
transformation requires full 3D position and velocity information, so we'll
have to specify a sky position, distance, proper motion components, and a
radial velocity. Let's start with coordinates in the ICRS frame::

    >>> icrs = coord.ICRS(ra=11.23*u.degree, dec=58.13*u.degree,
    ...                   distance=21.34*u.pc,
    ...                   pm_ra_cosdec=-55.89*u.mas/u.yr, pm_dec=71*u.mas/u.yr,
    ...                   radial_velocity=210.43*u.km/u.s)

We again use the ``transform_to()`` method to do the transformation::

    >>> icrs.transform_to(gc) # doctest: +FLOAT_CMP
    <Galactocentric Coordinate (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
        ( 266.4051, -28.936175)>, galcen_distance=8.0 kpc, galcen_v_sun=( 10.,  250.,  7.) km / s, z_sun=27.0 pc, roll=0.0 deg): (x, y, z) in pc
        (-8011.25186648,  18.02217595,  25.27812036)
     (v_x, v_y, v_z) in km / s
        (-97.06296832,  431.12942421, -2.69495881)>

Convert proper motions between the ``ICRS`` and ``Galactic`` frames
-------------------------------------------------------------------

The above Galactocentric coordinate transformations require full 3D position and
velocity information. However, transforming proper motion components between
different Barycentric coordinate frames is just a rotation, and can therefore be
done with just sky position and proper motions. For example, to convert from
ICRS proper motions to Galactic proper motions::

    >>> icrs = coord.ICRS(ra=11.23*u.degree, dec=58.13*u.degree,
    ...                   pm_ra_cosdec=-55.89*u.mas/u.yr, pm_dec=71*u.mas/u.yr)
    >>> gal = icrs.transform_to(coord.Galactic)
    >>> gal # doctest: +FLOAT_CMP
    <Galactic Coordinate: (l, b) in deg
        ( 122.06871373, -4.73082278)
     (pm_l_cosb, pm_b) in mas / yr
        (-54.0689922,  72.39638239)>
    >>> gal.pm_l_cosb # doctest: +FLOAT_CMP
    <Quantity -54.06899219513397 mas / yr>

.. _gala-coordinates-api:

API
===

.. automodapi:: gala.coordinates
    :no-inheritance-diagram:
    :skip: pm_gal_to_icrs
    :skip: pm_icrs_to_gal
    :skip: transform_proper_motion
    :skip: vgal_to_hel
    :skip: vgsr_to_vhel
    :skip: vhel_to_gal
    :skip: vhel_to_vgsr
    :skip: Quaternion
