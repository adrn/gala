# cython: language_level=3

# Standard-library
import warnings

# Third-party
import numpy as np
import astropy.units as u

# Project
from ..common import CommonBase
from ..potential import PotentialBase, CPotentialBase
from ..frame import FrameBase, CFrameBase, StaticFrame
from ...integrate import LeapfrogIntegrator, DOPRI853Integrator, Ruth4Integrator

__all__ = ["Hamiltonian"]


class Hamiltonian(CommonBase):
    """
    Represents a composition of a gravitational potential and a reference frame.

    This class is used to integrate orbits and compute quantities when working
    in non-inertial reference frames. The input potential and frame objects
    must have the same dimensionality and the same unit system. If both the
    potential and the frame are implemented in C, numerical orbit integration
    will use the C-implemented integrators and will be fast (to check if your
    object is C-enabled, check the ``.c_enabled`` attribute).

    Parameters
    ----------
    potential : :class:`~gala.potential.potential.PotentialBase` subclass
        The gravitational potential.
    frame : :class:`~gala.potential.frame.FrameBase` subclass (optional)
        The reference frame.

    """
    def __init__(self, potential, frame=None):
        if isinstance(potential, Hamiltonian):
            frame = potential.frame
            potential = potential.potential

        if frame is None:
            frame = StaticFrame(units=potential.units)

        elif not isinstance(frame, FrameBase):
            raise ValueError("Invalid input for reference frame. Must be a "
                             "FrameBase subclass.")

        if not isinstance(potential, PotentialBase):
            raise ValueError("Invalid input for potential. Must be a "
                             "PotentialBase subclass.")

        self.potential = potential
        self.frame = frame
        self._pot_ndim = self.potential.ndim
        self.ndim = 2 * self._pot_ndim

        if frame is not None:
            if frame.units != potential.units:
                raise ValueError(
                    "Potential and Frame must have compatible unit systems "
                    f"({potential.units} vs {frame.units})")

            if frame.ndim is not None and frame.ndim != potential.ndim:
                raise ValueError(
                    "Potential and Frame must have compatible phase-space "
                    f"dimensionality ({potential.ndim} vs {frame.ndim})")

        # TODO: document this attribute
        if isinstance(self.potential, CPotentialBase) and isinstance(self.frame, CFrameBase):
            self.c_enabled = True

        else:
            self.c_enabled = False

    @property
    def units(self):
        return self.potential.units

    def _energy(self, w, t):
        pot_E = self.potential._energy(np.ascontiguousarray(w[:, :self._pot_ndim]), t=t)
        other_E = self.frame._energy(w, t=t)
        return pot_E + other_E

    def _gradient(self, w, t):
        q = np.ascontiguousarray(w[:, :self._pot_ndim])

        dH = np.zeros_like(w)

        # extra terms from the frame
        dH += self.frame._gradient(w, t=t)
        dH[:, self._pot_ndim:] += self.potential._gradient(q, t=t)
        for i in range(self._pot_ndim):
            dH[:, self._pot_ndim+i] = -dH[:, self._pot_ndim+i]

        return dH

    def _hessian(self, w, t):
        raise NotImplementedError()

    # ========================================================================
    # Core methods that use the above implemented functions
    #
    def energy(self, w, t=0.):
        """
        Compute the energy (the value of the Hamiltonian) at the given phase-space position(s).

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
        w = self._remove_units_prepare_shape(w)
        orig_shape, w = self._get_c_valid_arr(w)
        t = self._validate_prepare_time(t, w)
        return self._energy(w, t=t).T.reshape(orig_shape[1:]) * self.units['energy'] / self.units['mass']

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
        w = self._remove_units_prepare_shape(w)
        orig_shape, w = self._get_c_valid_arr(w)
        t = self._validate_prepare_time(t, w)

        # TODO: wat do about units here?
        # ret_unit = self.units['length'] / self.units['time']**2
        return self._gradient(w, t=t).T.reshape(orig_shape)

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
        raise NotImplementedError()

    # def jacobi_energy(self, w, t=0.):
    #     """
    #     TODO: docstring
    #     TODO: if not rotating frame, raise error
    #     """

    #     if not isinstance(self.frame, gp.ConstantRotatingFrame):
    #         raise TypeError("The frame must be a ConstantRotatingFrame "
    #                         "to compute the Jacobi energy.")

    #     w = self._remove_units_prepare_shape(w)
    #     orig_shape, w = self._get_c_valid_arr(w)
    #     t = self._validate_prepare_time(t, w)

    #     E = self._energy(w, t=t).T.reshape(orig_shape[1:])
    #     L = np.cross(w[:, :3], w[:, 3:])

    #     Omega = self.frame.parameters['Omega']
    #     C = E - np.einsum('i, ...i->...', Omega, L).reshape(E.shape)
    #     return C * self.units['energy'] / self.units['mass']

    # ========================================================================
    # Python special methods
    #
    def __call__(self, w):
        return self.energy(w)

    # def __repr__(self):
    #     pars = ""
    #     keys = self.parameters.keys()

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

    #         pars += ("{}=" + par_fmt + post).format(k, v) + ", "

    #     if isinstance(self.units, DimensionlessUnitSystem):
    #         return "<{}: {} (dimensionless)>".format(self.__class__.__name__, pars.rstrip(", "))
    #     else:
    #         return "<{}: {} ({})>".format(self.__class__.__name__, pars.rstrip(", "), ",".join(map(str, self.units._core_units)))

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, other):
        return (self.potential == other.potential) and (self.frame == other.frame)

    def __ne__(self, other):
        return not self.__eq__(other)

    def integrate_orbit(self,
        w0,
        Integrator=None,
        Integrator_kwargs=dict(),
        cython_if_possible=True,
        store_all=True,
        **time_spec
    ):
        """
        Integrate an orbit in the current potential using the integrator class
        provided. Uses same time specification as `Integrator.run()` -- see
        the documentation for `gala.integrate` for more information.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        Integrator : `~gala.integrate.Integrator` (optional)
            Integrator class to use. By default, uses
            `~gala.integrate.LeapfrogIntegrator` if the frame is static and
            `~gala.integrate.DOPRI853Integrator` else.
        Integrator_kwargs : dict (optional)
            Any extra keyword argumets to pass to the integrator class
            when initializing. Only works in non-Cython mode.
        cython_if_possible : bool (optional)
            If there is a Cython version of the integrator implemented,
            and the potential object has a C instance, using Cython
            will be *much* faster.
        store_all : bool (optional)
            Controls whether to store the phase-space position at all intermediate
            timesteps. Set to False to store only the final values (i.e. the
            phase-space position(s) at the final timestep). Default is True.
        **time_spec
            Specification of how long to integrate. Most commonly, this is a
            timestep ``dt`` and number of steps ``n_steps``, or a timestep
            ``dt``, initial time ``t1``, and final time ``t2``. You may also
            pass in a time array with ``t``. See documentation for
            `~gala.integrate.parse_time_specification` for more information.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`

        """
        from gala.dynamics import PhaseSpacePosition, Orbit

        if Integrator is None and isinstance(self.frame, StaticFrame):
            Integrator = LeapfrogIntegrator
        elif Integrator is None:
            Integrator = DOPRI853Integrator
        else:
            # use the Integrator provided
            pass

        symplectic_integrators = [LeapfrogIntegrator, Ruth4Integrator]
        if (Integrator in symplectic_integrators and
                not isinstance(self.frame, StaticFrame)):
            warnings.warn(
                "Using a symplectic integrator with a non-static frame can "
                "lead to wildly incorrect orbits. It is recommended that you "
                "use DOPRI853Integrator instead.", RuntimeWarning)

        if not isinstance(w0, PhaseSpacePosition):
            w0 = np.asarray(w0)
            ndim = w0.shape[0]//2
            w0 = PhaseSpacePosition(pos=w0[:ndim], vel=w0[ndim:])

        ndim = w0.ndim
        arr_w0 = w0.w(self.units)
        arr_w0 = self._remove_units_prepare_shape(arr_w0)
        orig_shape, arr_w0 = self._get_c_valid_arr(arr_w0)

        if self.c_enabled and cython_if_possible:
            # array of times
            from ...integrate.timespec import parse_time_specification
            t = np.ascontiguousarray(parse_time_specification(self.units, **time_spec))

            # TODO: these replacements should be defined in gala.integrate...
            if Integrator == LeapfrogIntegrator:
                from ...integrate.cyintegrators import leapfrog_integrate_hamiltonian
                t, w = leapfrog_integrate_hamiltonian(self, arr_w0, t, store_all=store_all)

            elif Integrator == Ruth4Integrator:
                from ...integrate.cyintegrators import ruth4_integrate_hamiltonian
                t, w = ruth4_integrate_hamiltonian(self, arr_w0, t, store_all=store_all)

            elif Integrator == DOPRI853Integrator:
                from ...integrate.cyintegrators import dop853_integrate_hamiltonian
                t, w = dop853_integrate_hamiltonian(
                    self, arr_w0, t,
                    Integrator_kwargs.get('atol', 1E-10),
                    Integrator_kwargs.get('rtol', 1E-10),
                    Integrator_kwargs.get('nmax', 0),
                    Integrator_kwargs.get('progress', False),
                    store_all=store_all
                )
            else:
                raise ValueError(f"Cython integration not supported for '{Integrator!r}'")

            # because shape is different from normal integrator return
            w = np.rollaxis(w, -1)
            if w.shape[-1] == 1:
                w = w[..., 0]

        else:
            def F(t, w):
                # TODO: these Transposes are shitty and probably make it much slower?
                w_T = np.ascontiguousarray(w.T)
                return self._gradient(w_T, t=np.array([t])).T
            integrator = Integrator(F, func_units=self.units, **Integrator_kwargs)
            orbit = integrator.run(arr_w0.T, **time_spec)
            orbit.potential = self.potential
            orbit.frame = self.frame
            return orbit

        if not store_all:
            w = w[:, None]

        try:
            tunit = self.units['time']
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled

        return Orbit.from_w(w=w, units=self.units, t=t*tunit,
                            hamiltonian=self)

    # def save(self, f):
    #     """
    #     Save the potential to a text file. See :func:`~gala.potential.save`
    #     for more information.

    #     Parameters
    #     ----------
    #     f : str, file_like
    #         A filename or file-like object to write the input potential object to.

    #     """
    #     from .io import save
    #     save(self, f)
