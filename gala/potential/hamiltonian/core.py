# coding: utf-8

from __future__ import division, print_function

# Third-party
import numpy as np
import astropy.units as u

# Project
from ...integrate import LeapfrogIntegrator, DOPRI853Integrator
from ...dynamics import CartesianOrbit, CartesianPhaseSpacePosition
from ..potential import CPotentialBase
from ..frame import CFrameBase, StaticFrame

__all__ = ["Hamiltonian"]

class Hamiltonian(object):
    """
    TODO:
    """

    def __init__(self, potential, frame=None):

        # TODO: validate potential
        # TODO: validate frame

        if frame is None:
            frame = StaticFrame()

        self.potential = potential
        self.frame = frame
        self.n_dim = 2 * self.potential.n_dim

        # TODO: document this attribute
        if isinstance(self.potential, CPotentialBase) and isinstance(self.frame, CFrameBase):
            self.c_enabled = True
        else:
            self.c_enabled = False

    def _value(self, w, t=0.):
        # TODO: is this right?
        return self.potential._value(w[:self.n_dim]) + self.frame._value(w)

    def value(self, w, t=0.):
        """
        Compute the value of the Hamiltonian at the given phase-space position(s).

        Parameters
        ----------
        w : `~gala.dynamics.PhaseSpacePosition`, array_like
            The phase-space position to compute the value of the Hamiltonian.
            If the input object has no units (i.e. is an `~numpy.ndarray`), it
            is assumed to be in the same unit system as the potential class.

        Returns
        -------
        H : `~astropy.units.Quantity`
            Energy per unit mass or value of the Hamiltonian. If the input
            phase-space position has shape ``w.shape``, the output energy
            will have shape ``w.shape[1:]``.
        """
        # TODO: strip units from w
        # w = w...
        return self._value(w, t=t) * self.units['energy'] / self.units['mass']

    def _gradient(self, w, t=0.):

        grad = np.zeros_like(w)

        # extra terms from the frame
        grad += self.frame._gradient(w)

        # p_dot = -dH/dq
        grad[self.n_dim:] += -self.potential._gradient(w[:self.n_dim])

        return grad

    def gradient(self, w, t=0.):
        """
        Compute the gradient of the Hamiltonian at the given phase-space position(s).

        Parameters
        ----------
        w : `~gala.dynamics.PhaseSpacePosition`, array_like
            The phase-space position to compute the value of the Hamiltonian.
            If the input object has no units (i.e. is an `~numpy.ndarray`), it
            is assumed to be in the same unit system as the potential class.

        Returns
        -------
        TODO: this can't return a quantity, because units are different dH/dq vs. dH/dp
        grad : `~astropy.units.Quantity`
            The gradient of the potential. Will have the same shape as
            the input phase-space position, ``w``.
        """

        # TODO: strip units from w
        # w = w...
        return self._gradient(w, t=t) # TODO: see TODO about units about

    def _hessian(self, w, t=0.):
        raise NotImplementedError()

    def hessian(self, w, t=0.):
        """
        Compute the Hessian of the Hamiltonian at the given phase-space position(s).

        Parameters
        ----------
        w : `~gala.dynamics.PhaseSpacePosition`, array_like
            The phase-space position to compute the value of the Hamiltonian.
            If the input object has no units (i.e. is an `~numpy.ndarray`), it
            is assumed to be in the same unit system as the potential class.

        Returns
        -------
        # TODO: see TODO about units about
        hess : `~astropy.units.Quantity`
            The Hessian matrix of second derivatives of the potential. If the input
            position has shape ``w.shape``, the output energy will have shape
            ``(w.shape[0],w.shape[0]) + w.shape[1:]``. That is, an ``n_dim`` by
            ``n_dim`` array (matrix) for each position, where the dimensionality of
            phase-space is ``n_dim``.
        """
        # TODO: strip units from w
        # w = w...
        return self._hessian(w, t=t) # TODO: see TODO about units about

    # ========================================================================
    # Python special methods
    #
    def __call__(self, w):
        return self.value(w)

    # def __repr__(self):
    #     pars = ""
    #     if not isinstance(self.parameters, OrderedDict):
    #         keys = sorted(self.parameters.keys()) # to ensure the order is always the same
    #     else:
    #         keys = self.parameters.keys()

    #     for k in keys:
    #         v = self.parameters[k].value
    #         par_fmt = "{}"
    #         post = ""

    #         if hasattr(v,'unit'):
    #             post = " {}".format(v.unit)
    #             v = v.value

    #         if isinstance(v, float):
    #             if v == 0:
    #                 par_fmt = "{:.0f}"
    #             elif np.log10(v) < -2 or np.log10(v) > 5:
    #                 par_fmt = "{:.2e}"
    #             else:
    #                 par_fmt = "{:.2f}"

    #         elif isinstance(v, int) and np.log10(v) > 5:
    #             par_fmt = "{:.2e}"

    #         pars += ("{}=" + par_fmt + post).format(k,v) + ", "

    #     if isinstance(self.units, DimensionlessUnitSystem):
    #         return "<{}: {} (dimensionless)>".format(self.__class__.__name__, pars.rstrip(", "))
    #     else:
    #         return "<{}: {} ({})>".format(self.__class__.__name__, pars.rstrip(", "), ",".join(map(str, self.units._core_units)))

    def __str__(self):
        return self.__class__.__name__

    def integrate_orbit(self, w0, Integrator=LeapfrogIntegrator,
                        Integrator_kwargs=dict(), cython_if_possible=True,
                        **time_spec):
        """
        Integrate an orbit in the current potential using the integrator class
        provided. Uses same time specification as `Integrator.run()` -- see
        the documentation for `gala.integrate` for more information.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        Integrator : `~gala.integrate.Integrator` (optional)
            Integrator class to use.
        Integrator_kwargs : dict (optional)
            Any extra keyword argumets to pass to the integrator class
            when initializing. Only works in non-Cython mode.
        cython_if_possible : bool (optional)
            If there is a Cython version of the integrator implemented,
            and the potential object has a C instance, using Cython
            will be *much* faster.
        **time_spec
            Specification of how long to integrate. See documentation
            for `~gala.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.CartesianOrbit`

        """

        if not isinstance(w0, CartesianPhaseSpacePosition):
            w0 = np.asarray(w0)
            ndim = w0.shape[0]//2
            w0 = CartesianPhaseSpacePosition(pos=w0[:ndim],
                                             vel=w0[ndim:])

        ndim = w0.ndim
        arr_w0 = w0.w(self.units)
        if hasattr(self, 'c_instance') and cython_if_possible:
            # WARNING TO SELF: this transpose is there because the Cython
            #   functions expect a shape: (norbits, ndim)
            arr_w0 = np.ascontiguousarray(arr_w0.T)

            # array of times
            from ..integrate.timespec import parse_time_specification
            t = np.ascontiguousarray(parse_time_specification(self.units, **time_spec))

            if Integrator == LeapfrogIntegrator:
                from ..integrate.cyintegrators import leapfrog_integrate_hamiltonian
                t,w = leapfrog_integrate_hamiltonian(self.c_instance, arr_w0, t)

            elif Integrator == DOPRI853Integrator:
                from ..integrate.cyintegrators import dop853_integrate_hamiltonian
                t,w = dop853_integrate_hamiltonian(self.c_instance, arr_w0, t,
                                                   Integrator_kwargs.get('atol', 1E-10),
                                                   Integrator_kwargs.get('rtol', 1E-10),
                                                   Integrator_kwargs.get('nmax', 0))
            else:
                raise ValueError("Cython integration not supported for '{}'".format(Integrator))

            # because shape is different from normal integrator return
            w = np.rollaxis(w, -1)
            if w.shape[-1] == 1:
                w = w[...,0]

        else:
            def acc(t, w):
                return np.vstack((w[ndim:], -self._gradient(w[:ndim], t=t)))
            integrator = Integrator(acc, func_units=self.units, **Integrator_kwargs)
            orbit = integrator.run(w0, **time_spec)
            orbit.potential = self
            return orbit

        try:
            tunit = self.units['time']
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled
        return CartesianOrbit.from_w(w=w, units=self.units, t=t*tunit, potential=self)

    def save(self, f):
        """
        Save the potential to a text file. See :func:`~gala.potential.save`
        for more information.

        Parameters
        ----------
        f : str, file_like
            A filename or file-like object to write the input potential object to.

        """
        from .io import save
        save(self, f)
