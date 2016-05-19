.. _streamframes:

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import gala.coordinates as gc

Stellar Stream Coordinate Frames
================================

Also included in this subpackage are Astropy coordinate frame classes for
transforming to Sagittarius and Orphan stream coordinates (as defined in the
references below). These classes behave like the built-in astropy coordinates
frames (e.g., :class:`~astropy.coordinates.ICRS` or
:class:`~astropy.coordinates.Galactic`) and can be transformed to and from
other astropy coordinate frames::

    >>> c = coord.SkyCoord(ra=100.68458*u.degree, dec=41.26917*u.degree)
    >>> c.transform_to(gc.Sagittarius) # doctest: +FLOAT_CMP
    <SkyCoord (Sagittarius): (Lambda, Beta) in deg
        (179.58511054, -12.55845019)>
    >>> s = gc.Sagittarius(Lambda=156.342*u.degree, Beta=1.1*u.degree)
    >>> c = coord.SkyCoord(s)
    >>> c.galactic # doctest: +FLOAT_CMP
    <SkyCoord (Galactic): (l, b) in deg
        (182.5922090437946, -9.539692094685897)>

References
----------

* `A 2MASS All-Sky View of the Sagittarius Dwarf Galaxy: I. Morphology of the Sagittarius Core and Tidal Arms <http://arxiv.org/abs/astro-ph/0304198>`_
* `The Orbit of the Orphan Stream <http://arxiv.org/abs/1001.0576>`_
