"""
A standalone orchestrator (modeled on ``DirectNBody``) that integrates orbits
with an added stochastic velocity-diffusion term, using a fixed-step
Euler-Maruyama scheme implemented in Cython.
"""

import astropy.units as u
import numpy as np

from ...integrate.timespec import parse_time_specification
from ...potential import Hamiltonian, StaticFrame
from ...units import UnitSystem
from ...util import atleast_2d
from ..core import PhaseSpacePosition
from ..orbit import Orbit
from .cydiffusion import euler_maruyama_integrate
from .diffusion_models import DiffusionBase

__all__ = ["StochasticOrbitIntegrator"]


class StochasticOrbitIntegrator:
    """Integrate tracer orbits with stochastic (diffusive) velocity kicks.

    This is a trial implementation of an SDE integrator using a fixed-step
    Euler-Maruyama scheme: each step, positions and velocities are advanced by
    the deterministic (static-frame) dynamics and a stochastic velocity kick set
    by a diffusion model is added. With a zero diffusion model this reduces to
    deterministic forward-Euler integration. This is a low-order scheme; a
    higher-order scheme may be added in the future.

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase` subclass instance
        The (C-enabled) background potential the orbits are integrated in.
    diffusion : `~gala.dynamics.diffusion.DiffusionBase` subclass instance
        The velocity-space diffusion model.
    frame : `~gala.potential.frame.FrameBase` subclass (optional)
        The reference frame. Only static frames are currently supported.
    units : `~gala.units.UnitSystem` (optional)
        The unit system to work in. If not provided, taken from the potential.
    seed : int (optional)
        Seed for the random number generator, for reproducibility. If not
        provided, a random seed is drawn.
    """

    def __init__(self, potential, diffusion, frame=None, units=None, seed=None):
        if units is None:
            units = getattr(potential, "units", None)
        if units is None:
            units = getattr(diffusion, "units", None)
        if units is None:
            raise ValueError(
                "Could not determine units from input! Pass in a unit system "
                "with `units`, or use a potential with a valid unit system."
            )
        if not isinstance(units, UnitSystem):
            units = UnitSystem(units)
        self.units = units

        if not isinstance(diffusion, DiffusionBase):
            raise TypeError(
                "`diffusion` must be a "
                "gala.dynamics.diffusion.DiffusionBase instance, not "
                f"'{type(diffusion).__name__}'"
            )
        if diffusion.units != self.units:
            diffusion = diffusion.replace_units(self.units)
        self.diffusion = diffusion

        if getattr(potential, "units", None) != self.units:
            potential = potential.replace_units(self.units)
        self.potential = potential

        if frame is None:
            frame = StaticFrame(self.units)
        elif not isinstance(frame, StaticFrame):
            raise ValueError(
                "The diffusion integrator currently only supports static "
                f"frames, not '{type(frame).__name__}'."
            )
        self.frame = frame

        self.H = Hamiltonian(self.potential, frame=self.frame)
        if not self.H.c_enabled:
            raise ValueError(
                "Input potential must be C-enabled: one or more components in "
                "the input potential are Python-only."
            )

        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**63 - 1))
        self.seed = int(seed)

    def __repr__(self):
        return (
            f"<StochasticOrbitIntegrator potential={self.potential!r}, "
            f"diffusion={self.diffusion!r}>"
        )

    def integrate_orbit(self, w0, save_all=True, **time_spec):
        """Integrate the initial conditions with stochastic velocity kicks.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        save_all : bool (optional)
            Save the phase-space position at all timesteps. If False, only the
            final state is returned.
        **time_spec
            Time specification, e.g. ``dt`` and ``n_steps``. Must produce a
            fixed timestep. See
            `~gala.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`
        """
        if isinstance(w0, PhaseSpacePosition):
            arr_w0 = w0.w(self.units)
        else:
            arr_w0 = np.asarray(w0)
        arr_w0 = atleast_2d(arr_w0, insert_axis=1)
        arr_w0 = np.ascontiguousarray(arr_w0, dtype=np.float64)

        t = np.ascontiguousarray(parse_time_specification(self.units, **time_spec))

        t, w = euler_maruyama_integrate(
            self.H, arr_w0, t, self.diffusion, self.seed, int(save_all)
        )

        if w.shape[-1] == 1:
            w = w[..., 0]
        if not save_all:
            w = w[:, None]

        try:
            tunit = self.units["time"]
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled
        t = u.Quantity(t, tunit, copy=False)

        return Orbit.from_w(w=w, units=self.units, t=t, hamiltonian=self.H)
