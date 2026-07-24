"""
Python-facing diffusion-coefficient model classes for the trial SDE integrator.

Each class declares its parameters with :class:`~gala.potential.common.PotentialParameter`
(reusing gala's parameter/unit machinery) and links to a compiled Cython
``*Wrapper`` that points at a C diffusion model function.

Unit convention
---------------
Diffusion coefficients have units of ``velocity**2 / time`` (a diffusion tensor)
which is not a named astropy physical type, so parameters are handled directly:
pass an astropy :class:`~astropy.units.Quantity` (decomposed into the model's unit
system) or a plain number already expressed in that unit system. For example, in
``galactic`` units a diffusion rate is in ``(kpc/Myr)**2 / Myr``.
"""

import numpy as np

from ...potential.common import CommonBase, PotentialParameter
from .cydiffusion import (
    ConstantDiagDiffusionWrapper,
    ConstantTensorDiffusionWrapper,
    ExampleRadialDiffusionWrapper,
)

__all__ = [
    "ConstantDiffusion",
    "ConstantTensorDiffusion",
    "DiffusionBase",
    "ExampleRadialDiffusion",
]


class DiffusionBase(CommonBase):
    """Base class for velocity-space diffusion models.

    Subclasses declare parameters as class-level
    :class:`~gala.potential.common.PotentialParameter` descriptors and set the
    ``Wrapper`` attribute to the corresponding Cython wrapper class.
    """

    ndim = 3
    Wrapper = None
    # expected length of the flattened C parameter array (None to skip the check)
    _n_c_params = None

    def __init__(self, *args, units=None, **kwargs):
        parameter_values, parameter_is_default = self._parse_parameter_values(
            *args, **kwargs
        )
        self._units = self._validate_units(units)
        self.parameters = parameter_values
        self.parameter_is_default = set(parameter_is_default)
        self._setup_wrapper()

    @property
    def units(self):
        return self._units

    def _param_value_in_units(self, v):
        if hasattr(v, "unit"):
            return np.atleast_1d(v.decompose(self.units).value).ravel()
        return np.atleast_1d(np.asarray(v, dtype=float)).ravel()

    def _setup_wrapper(self):
        if self.Wrapper is None:
            raise ValueError(
                f"Diffusion wrapper class not defined for {self.__class__}"
            )

        arrs = []
        for k, v in self.parameters.items():
            if self._parameters[k].python_only:
                continue
            arrs.append(self._param_value_in_units(v))

        c_parameters = np.concatenate(arrs) if arrs else np.array([], dtype=float)

        if self._n_c_params is not None and len(c_parameters) != self._n_c_params:
            raise ValueError(
                f"{self.__class__.__name__} expected {self._n_c_params} C "
                f"parameter value(s) but got {len(c_parameters)}. Check the "
                "shape of the parameter(s) you passed in."
            )

        self.c_parameters = np.ascontiguousarray(c_parameters, dtype=np.float64)
        self.c_instance = self.Wrapper(self.c_parameters, self.ndim)

    def replace_units(self, units):
        """Return a copy of this diffusion model in a new unit system."""
        params = {
            k: v
            for k, v in self.parameters.items()
            if k not in self.parameter_is_default
        }
        return self.__class__(**params, units=units)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {dict(self.parameters)}>"


class ConstantDiffusion(DiffusionBase):
    r"""Constant, diagonal velocity-space diffusion.

    Each velocity component :math:`i` diffuses independently with rate
    :math:`D_i`, so that the per-step velocity kick has covariance
    :math:`\mathrm{diag}(D)\,\Delta t`.

    Parameters
    ----------
    D : array_like or `~astropy.units.Quantity`
        Length-``ndim`` (=3) vector of per-component diffusion rates, in units
        of ``velocity**2 / time``.
    units : `~gala.units.UnitSystem`
        The unit system the coefficients are expressed in.
    """

    D = PotentialParameter("D", physical_type=None, ndim=1)
    Wrapper = ConstantDiagDiffusionWrapper
    _n_c_params = 3


class ConstantTensorDiffusion(DiffusionBase):
    r"""Constant, full-tensor velocity-space diffusion.

    The per-step velocity kick has covariance :math:`D\,\Delta t`, where
    :math:`D` is a symmetric positive-semidefinite ``ndim x ndim`` diffusion
    tensor.

    Parameters
    ----------
    D : array_like or `~astropy.units.Quantity`
        Symmetric ``ndim x ndim`` (=3x3) diffusion tensor, in units of
        ``velocity**2 / time``.
    units : `~gala.units.UnitSystem`
        The unit system the coefficients are expressed in.
    """

    D = PotentialParameter("D", physical_type=None, ndim=2)
    Wrapper = ConstantTensorDiffusionWrapper
    _n_c_params = 9


class ExampleRadialDiffusion(DiffusionBase):
    r"""Example position-dependent diagonal diffusion (a TEMPLATE).

    Demonstrates a model whose amplitude depends on position: the diagonal
    diffusion factor is scaled by :math:`\exp(-|q| / r_s)`. This is an
    illustration of the pattern for adding real, spatially varying prescriptions
    (e.g. ISM or impulsive-kick diffusion) -- it is not a physical model.

    Parameters
    ----------
    D : array_like or `~astropy.units.Quantity`
        Length-``ndim`` (=3) vector of central per-component diffusion rates, in
        units of ``velocity**2 / time``.
    r_s : `~astropy.units.Quantity` or float
        Scale radius over which the diffusion amplitude decays, in length units.
    units : `~gala.units.UnitSystem`
        The unit system the coefficients are expressed in.
    """

    D = PotentialParameter("D", physical_type=None, ndim=1)
    r_s = PotentialParameter("r_s", physical_type="length", ndim=0)
    Wrapper = ExampleRadialDiffusionWrapper
    _n_c_params = 4
