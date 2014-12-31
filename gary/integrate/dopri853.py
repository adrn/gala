# coding: utf-8

""" Wrapper around SciPy DOPRI853 integrator. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from scipy.integrate import ode

# Project
from .core import Integrator
from .timespec import _parse_time_specification

__all__ = ["DOPRI853Integrator"]

class DOPRI853Integrator(Integrator):
    r"""
    This provides a wrapper around `Scipy`'s implementation of the
    Dormand-Prince 85(3) integration scheme.

    .. seealso::

        - Numerical recipes (Dopr853)
        - http://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

    **Example:** Harmonic oscillator

    Hamilton's equations are

    .. math::

        \dot{q} = \frac{\partial H}{\partial p}\\
        \dot{p} = -\frac{\partial H}{\partial q}

    The harmonic oscillator Hamiltonian is

    .. math::

        H(q,p) = \frac{1}{2}(p^2 + q^2)

    so that the equations of motion are given by

    .. math::

        \dot{q} = p\\
        \dot{p} = -q

    We then define a vector :math:`x = (q, p)`. The function passed in to
    the integrator should return the derivative of :math:`x` with respect to
    the independent variable,  :math:`\dot{x} = (\dot{q}, \dot{p})`, e.g.::

        def F(t,x):
            q,p = x.T
            return np.array([p,-q]).T

    To create an integrator object, just pass this function in to the
    constructor, and then we can integrate orbits from a given vector of
    initial conditions::

        integrator = DOPRI853Integrator(F)
        times,ws = integrator.run(w0=[1.,0.], dt=0.1, nsteps=1000)

    .. note::

        Even though we only pass in a single vector of initial conditions,
        this gets promoted internally to a 2D array. This means the shape of
        the integrated orbit array will always be 3D. In this case, `ws` will
        have shape `(1001,1,2)`.

    Parameters
    ----------
    func : func
        A callable object that computes the phase-space coordinate
        derivatives with respect to the independent variable at a point
        in phase space.
    func_args : tuple (optional)
        Any extra arguments for the function.

    """

    def __init__(self, func, func_args=(), **kwargs):

        if not hasattr(func, '__call__'):
            raise ValueError("func must be a callable object, e.g., a function.")

        self.func = func
        self._func_args = func_args
        self._ode_kwargs = kwargs

    def run(self, w0, mmap=None, **time_spec):
        """
        Run the integrator starting at the given coordinates and momenta
        (or velocities) and a time specification. The initial conditions
        `w0` should have shape `(nparticles, ndim)` or `(ndim,)` for a
        single orbit.

        There are a few combinations of keyword arguments accepted for
        specifying the timestepping. For example, you can specify a fixed
        timestep (`dt`) and a number of steps (`nsteps`), or an array of
        times. See **Other Parameters** below for more information.

        Parameters
        ==========
        w0 : array_like
            Initial conditions.

        Other Parameters
        ================
        dt, nsteps[, t1] : (numeric, int[, numeric])
            A fixed timestep dt and a number of steps to run for.
        dt, t1, t2 : (numeric, numeric, numeric)
            A fixed timestep dt, an initial time, and a final time.
        t : array_like
            An array of times to solve on.

        Returns
        =======
        times : array_like
            An array of times.
        w : array_like
            The array of positions and momenta (velocities) at each time in
            the time array. This array has shape `(Ntimes,Norbits,Ndim)`.

        """

        # generate the array of times
        times = _parse_time_specification(**time_spec)
        nsteps = len(times)-1

        w0, ws = self._prepare_ws(w0, mmap, nsteps)
        nparticles, ndim = w0.shape

        # need this to do resizing, and to handle func_args because there is some
        #   issue with the args stuff in scipy...
        def func_wrapper(t,x):
            _x = x.reshape((nparticles,ndim))
            return self.func(t,_x,*self._func_args).reshape((nparticles*ndim,))

        self._ode = ode(func_wrapper, jac=None)
        self._ode = self._ode.set_integrator('dop853', **self._ode_kwargs)

        # create the return arrays
        ws[0] = w0

        # make 1D
        w0 = w0.reshape((nparticles*ndim,))

        # set the initial conditions
        self._ode.set_initial_value(w0, times[0])

        # Integrate the ODE(s) across each delta_t timestep
        k = 1
        while self._ode.successful() and k < (nsteps+1):
            self._ode.integrate(times[k])
            outy = self._ode.y
            ws[k] = outy.reshape(nparticles,ndim)
            k += 1

        if not self._ode.successful():
            raise RuntimeError("ODE integration failed!")

        return times, ws
