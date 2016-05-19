.. _propermotion:

    >>> import numpy as np
    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import gala.coordinates as gc

Proper motion transformations
=============================

Transforming between ICRS and Galactic proper motions is supported in Gala. To
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
