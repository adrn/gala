__all__ = ["FrameBase"]


from ..common import CommonBase


class FrameBase(CommonBase):
    ndim = 3

    def __init__(self, *args, units=None, **kwargs):
        parameter_values, parameter_is_default = self._parse_parameter_values(
            *args, **kwargs
        )
        self._setup_frame(
            parameters=parameter_values,
            parameter_is_default=parameter_is_default,
            units=units,
        )

    def _setup_frame(self, parameters, parameter_is_default, units=None):
        self.units = self._validate_units(units)
        self.parameters = self._prepare_parameters(parameters, self.units)
        self.parameter_is_default = set(parameter_is_default)
