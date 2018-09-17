.. _greatcircle:

*************************************************
Great circle and stellar stream coordinate frames
*************************************************

We'll assume the following imports have already been executed::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> import gala.dynamics as gd
    >>> from astropy.coordinates import (CylindricalRepresentation,
    ...                                  CylindricalDifferential)
    >>> from gala.units import galactic
    >>> np.random.seed(42)

Introduction
============

Great circle coordinate systems are defined as a rotation from another spherical
coordinate system, such as the ICRS. The resulting system is aligned so that the
north pole is the pole of a specified great circle, with an additional
specification of the longitude zero point of the final system.

Currently, ``gala`` only supports great circle frames that are defined as a
rotation away from the ICRS (ra, dec) through the
`~gala.coordinates.GreatCircleICRSFrame` class. To create a new great circle frame, you must specify a pole, and optionally specify the longitude zero point either by specifying the right ascension of the longitude zero point, or by specifying a final rotation to be applied to the transformation. For example,
to create ...::

    >>> pole = coord.SkyCoord(ra=32.5423*u.deg, dec=19.8813*u.deg)
    >>> fr = gc.GreatCircleICRSFrame(pole=pole)
    >>> c = coord.SkyCoord(ra=160*u.deg, dec=-11*u.deg)
    >>> c.transform_to(fr)
    <SkyCoord (GreatCircleICRSFrame: pole=<ICRS Coordinate: (ra, dec) in deg
    ( 32.5423,  19.8813)>, ra0=nan deg, rotation=0.0 deg): (phi1, phi2) in deg
    ( 122.50597343, -22.48617839)>




.. _greatcircle-api:

API
===

.. automodapi:: gala.coordinates.greatcircle
    :no-inheritance-diagram:
