__all__ = ['FrameBase']

# This package
from ..common import CommonBase


class FrameBase(CommonBase):
    ndim = 3

    def __init__(self, *args, units=None, **kwargs):
        parameter_values = self._parse_parameter_values(*args, **kwargs)
        self._setup_frame(parameters=parameter_values,
                          units=units)

    def _setup_frame(self, parameters, units=None):
        self.units = self._validate_units(units)
        self.parameters = self._prepare_parameters(parameters, self.units)
