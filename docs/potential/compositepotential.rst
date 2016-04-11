.. _compositepotential:

************************************
Creating a multi-component potential
************************************

Potential objects can be combined into more complex *composite* potentials
using the :class:`~gary.potential.CompositePotential` class. This
class operates like a Python dictionary in that each component potential
must be named, and the potentials can either be passed in to the initializer
or added after the composite potential container is already created. Either
way, each component potential must be instantiated before adding it to the
composite potential::

    >>> import numpy as np
    >>> import gary.potential as gp
    >>> from gary.units import galactic
    >>> disk = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    >>> bulge = gp.HernquistPotential(m=3E10, c=0.7, units=galactic)
    >>> pot = gp.CompositePotential(disk=disk, bulge=bulge)

is equivalent to::

    >>> pot = gp.CompositePotential()
    >>> pot['disk'] = disk
    >>> pot['bulge'] = bulge

The resulting potential object has all of the same properties as individual
potential objects::

    >>> pot.value([1.,-1.,0.]) # doctest: +FLOAT_CMP
    <Quantity [-0.12891172] kpc2 / Myr2>
    >>> pot.acceleration([1.,-1.,0.]) # doctest: +FLOAT_CMP
    <Quantity [[-0.02271507],
               [ 0.02271507],
               [-0.        ]] kpc / Myr2>
    >>> grid = np.linspace(-3.,3.,100)
    >>> fig = pot.plot_contours(grid=(grid,0,grid))

.. plot::
    :align: center

    import numpy as np
    import gary.dynamics as gd
    import gary.potential as gp
    from gary.units import galactic

    disk = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    bulge = gp.HernquistPotential(m=3E10, c=0.7, units=galactic)
    pot = gp.CompositePotential(disk=disk, bulge=bulge)

    grid = np.linspace(-3.,3.,100)
    fig = pot.plot_contours(grid=(grid,0,grid))
