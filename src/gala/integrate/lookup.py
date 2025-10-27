"""Lookup utilities for integrator classes."""

from .pyintegrators import (
    DOPRI853Integrator,
    LeapfrogIntegrator,
    RK5Integrator,
    Ruth4Integrator,
)

__all__ = ["get_integrator"]


# Mapping from lowercase integrator names to integrator classes
_integrator_name_mapping = {
    "leapfrog": LeapfrogIntegrator,
    "dopri853": DOPRI853Integrator,
    "dop853": DOPRI853Integrator,
    "rk5": RK5Integrator,
    "ruth4": Ruth4Integrator,
}


def get_integrator(name):
    """Get an integrator class from a string name.

    This function allows you to specify an integrator by its string name instead
    of directly importing the class.

    Parameters
    ----------
    name : str or Integrator class
        The name of the integrator (case-insensitive) or an integrator class.
        If an integrator class is passed in, it is returned unchanged. Valid
        integrator names are: 'leapfrog', 'dopri853', 'dop853', 'rk5', 'ruth4'.

    Returns
    -------
    integrator_cls : Integrator class
        The integrator class corresponding to the input name.

    Examples
    --------
    Get an integrator class by name::

        >>> from gala.integrate import get_integrator
        >>> LeapfrogIntegrator = get_integrator('leapfrog')
        >>> LeapfrogIntegrator
        <class 'gala.integrate.pyintegrators.leapfrog.LeapfrogIntegrator'>

    Use with potential integration::

        >>> import gala.potential as gp
        >>> pot = gp.HernquistPotential(m=1e11, c=10, units='galactic')
        >>> w0 = gd.PhaseSpacePosition(pos=[10,0,0], vel=[0,175,0])
        >>> orbit = gp.Hamiltonian(pot).integrate_orbit(
        ...     w0, dt=1., n_steps=1000, Integrator='leapfrog'
        ... )

    If you pass in an integrator class, it is returned unchanged::

        >>> from gala.integrate import LeapfrogIntegrator
        >>> get_integrator(LeapfrogIntegrator) is LeapfrogIntegrator
        True

    """
    # If it's already an integrator class, return it
    if not isinstance(name, str):
        if name not in list(_integrator_name_mapping.values()):
            raise ValueError(
                f"Integrator class '{name}' is not recognized. Valid classes are: "
                f"{list(_integrator_name_mapping.values())}"
            )
        return name

    # Convert to lowercase for case-insensitive lookup
    name_lower = name.lower()

    try:
        return _integrator_name_mapping[name_lower]
    except KeyError as e:
        raise ValueError(
            f"Integrator name '{name}' is not recognized. Valid names are: "
            f"{list(_integrator_name_mapping.keys())}"
        ) from e
