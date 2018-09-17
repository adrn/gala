.. _coordinates:

*********************************************
Coordinate Systems (`gala.coordinates`)
*********************************************

Introduction
============

The `~gala.coordinates` subpackage primarily provides specialty
:mod:`astropy.coordinates` frame classes for coordinate systems defined by the
stellar streams, and for other common Galactic dynamics tasks like removing
solar reflex motion from proper motions or radial velocities.

For the examples below the following imports have already been executed::

    >>> import numpy as np
    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import gala.coordinates as gc

Stellar stream coordinate frames
================================

`Gala` provides Astropy coordinate frame classes for transforming to several
built-in stellar stream stream coordinate frames (as defined in the references
below), and for transforming positions and velocities to and from coordinate
systems defined by great circles or poles. These classes behave like the
built-in astropy coordinates frames (e.g., :class:`~astropy.coordinates.ICRS` or
:class:`~astropy.coordinates.Galactic`) and can be transformed to and from other
astropy coordinate frames. For example, to convert a set of
`~astropy.coordinates.ICRS` (RA, Dec) coordinates to a coordinate system aligned
with the Sagittarius stream with the `~gala.coordinates.Sagittarius` frame::

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
        (181.28968151, 54.84972806)
     (pm_l_cosb, pm_b) in mas / yr
        (12.03209393, -3.69847479)>

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
        (266.4051, -28.936175)>, galcen_distance=8.3 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg): (x, y, z) in kpc
        (-12.78977138, -0.09870921, 6.44110283)
     (v_x, v_y, v_z) in km / s
        (-73.01933674, -216.37648654, -97.60065189)>

For custom great circle coordinate systems, and for more information about the
stellar stream frames, see :ref:`greatcircle`.


Correcting velocities for solar reflex motion
---------------------------------------------

The `~gala.coordinates.reflex_correct` function accepts an Astropy
`~astropy.coordinates.SkyCoord` object with position and velocity information,
and returns a coordinate object with the solar motion added back in to the
velocity components. This is useful for computing velocities in a Galactocentric
reference frame, rather than a solar system barycentric frame.

To use, you pass in a coordinate object with scalar or array values::

    >>> c = coord.SkyCoord(ra=[180.323, 1.523]*u.deg,
    ...                    dec=[-17, 29]*u.deg,
    ...                    distance=[172, 412]*u.pc,
    ...                    pm_ra_cosdec=[-11, 3]*u.mas/u.yr,
    ...                    pm_dec=[4, 8]*u.mas/u.yr,
    ...                    radial_velocity=[114, -21]*u.km/u.s)
    >>> gc.reflex_correct(c) # doctest: +FLOAT_CMP
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        [(180.323, -17., 172.), (  1.523,  29., 412.)]
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        [(130.07952506, 166.6468671 , -38.61764069),
         (-57.03358538,  58.77444739, 153.72743857)]>

By default, this uses the default solar location and velocity from the
`astropy.coordinates.Galactocentric` frame class. To modify these, for example,
to change the solar velocity, or the sun's height above the Galactic midplane,
use the arguments of that class::

    >>> vsun = coord.CartesianDifferential([11., 245., 7.]*u.km/u.s)
    >>> gc_frame = coord.Galactocentric(galcen_v_sun=vsun, z_sun=0*u.pc)
    >>> gc.reflex_correct(c, gc_frame) # doctest: +FLOAT_CMP
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        [(180.323, -17., 172.), (  1.523,  29., 412.)]
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        [(136.93481249, 175.37627916, -47.6177433 ),
         (-59.96484921,  61.41044742, 163.90707073)]>

If you don't have radial velocity information and want to correct the proper
motions, pass in zeros for the radial velocity (and ignore the output value of
the radial velocity)::

    >>> c = coord.SkyCoord(ra=162*u.deg,
    ...                    dec=-17*u.deg,
    ...                    distance=172*u.pc,
    ...                    pm_ra_cosdec=-11*u.mas/u.yr,
    ...                    pm_dec=4*u.mas/u.yr,
    ...                    radial_velocity=0*u.km/u.s)
    >>> gc.reflex_correct(c) # doctest: +FLOAT_CMP
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        (162., -17., 172.)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (81.60359171, 155.62817259, -182.00367694)>

Similarly, if you don't have proper motion information and want to correct the
proper motions, pass in zeros for the proper motions (and ignore the output
values of the proper motions) -- this is sometimes called "v_GSR"::

    >>> c = coord.SkyCoord(ra=162*u.deg,
    ...                    dec=-17*u.deg,
    ...                    distance=172*u.pc,
    ...                    pm_ra_cosdec=0*u.mas/u.yr,
    ...                    pm_dec=0*u.mas/u.yr,
    ...                    radial_velocity=127*u.km/u.s)
    >>> gc.reflex_correct(c) # doctest: +FLOAT_CMP
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        (162., -17., 172.)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (92.60359171, 151.62817259, -55.00367694)>


References
----------

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the
  Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_
* `Constraining the Milky Way potential with a 6-D phase-space map of the GD-1
  stellar stream <https://arxiv.org/abs/0907.1085>`_


Using gala.coordinates
======================

More details are provided in the linked pages below:

.. toctree::
   :maxdepth: 1

   greatcircle


.. _gala-coordinates-api:

API
===

.. automodapi:: gala.coordinates
    :no-inheritance-diagram:
    :skip: vgsr_to_vhel
    :skip: vhel_to_vgsr
    :skip: Quaternion
