__all__ = ['FrameBase']

# This package
from ..common import CommonBase


class FrameBase(CommonBase):
    ndim = 3

    def __init__(self, units=None, **kwargs):
        parameter_values = dict()
        for k in self._parameters:
            val = kwargs.pop(k, self._parameters[k].default)
            parameter_values[k] = val

        if len(kwargs):
            raise ValueError(f"{self.__class__} received unexpected keyword "
                             f"argument(s): {list(kwargs.keys())}")

        self._setup_frame(parameters=parameter_values,
                          units=units)

    def _setup_frame(self, parameters, units=None):
        self.units = self._validate_units(units)
        self.parameters = self._prepare_parameters(parameters, self.units)
