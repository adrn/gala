.. _custompotential:

*********************************
Defining your own potential class
*********************************

Introduction
============

There are two ways to define a new potential class: with pure-Python, or with
Cython. The advantage to writing a new class in Cython is that the
computations can execute with C-like speeds, however only certain integrators
support using this functionality anyways. If you are not familiar with Cython,
you probably want to stick to a pure-Python class.

Adding a custom potential with Python
-------------------------------------

New potentials are implemented by subclassing one of the potential base
classes and defining functions that compute the value and gradient of the
potential. We will work through an example below for adding a
`Henon-Heiles potential <http://en.wikipedia.org/wiki/H%C3%A9non-Heiles_System>`_.
We start by importing the necessary subpackage::

    import gary.potential as gp

For most potential classes, the user must supply a unit system when defining
the potential. For this reason, we must wrap the value and gradient function
definitions *within another function*. This looks a bit funny at first glance,
but is necessary for potentials that require, for instance, the gravitational
constant, which must be in the correct unit system to work properly.

However, the Henon-Heiles potential can be expressed as a dimensionless
function. We must still wrap our function definitions in a separate function,
but we just won't use the units passed in::

    def henon_heiles_funcs(units):

        def value(r, L):
            x,y = r.T
            return 0.5*(x**2 + y**2) + L*(x**2*y - y**3/3)

        def gradient(r, L):
            x,y = r.T
            grad = np.zeros_like(r)
            grad[...,0] = x + 2*L*x*y
            grad[...,1] = y + L*(x**2 - y**2)
            return grad

        def hessian(r, L):
            raise NotImplementedError()

        return value, gradient, hessian

For this example, we won't implement a Hessian function, but we still must
define and return a function. We now define a subclass for this potential::

    class HenonHeilesPotential(gp.CartesianPotential):
        r"""
        The Henon-Heiles potential originally used to describe the non-linear
        motion of stars near the Galactic center.

        .. math::

            \Phi = \frac{1}{2}(x^2 + y^2) + \lambda(x^2 y - \frac{y^3}{3})

        Parameters
        ----------
        L : numeric
            Lambda parameter.
        units : iterable
            Unique list of non-reducable units that specify (at minimum) the
            length, mass, time, and angle units.
        """

        def __init__(self, L, units=None):
            parameters = dict(L=L)
            func,gradient,hessian = henon_heiles_funcs(units)
            super(HenonHeilesPotential, self).__init__(func=func, gradient=gradient,
                                                       hessian=hessian,
                                                       parameters=parameters, units=units)

That's it! Most of the above code is just documentation for this new potential,
and now we can create instances of the object which have all of the features of
the other potential classes. For example, we can integrate an orbit::

    potential = HenonHeilesPotential(L=0.5)
    t,w = potential.integrate_orbit([0.,0.,0.5,0.5], dt=0.03, nsteps=50000)

Or create a contour plot::

    grid = np.linspace(-2,2,100)
    fig = potential.plot_contours(grid=(grid,grid),
                                  levels=[0, 0.05,0.1,1/6.,0.5,1.,2,3,5],
                                  cmap='Blues_r', subplots_kw=dict(figsize=(6,6)),
                                  labels=['$x$','$y$'])
    fig.axes[0].plot(w[:,0,0], w[:,0,1], marker='.',
                     linestyle='none', color='#fec44f', alpha=0.1)

.. image:: ../_static/potential/henon-heiles.png

Adding a custom potential with Cython
-------------------------------------

TODO:
