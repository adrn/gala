from astropy.coordinates import Attribute


class StringValidatedAttribute(Attribute):
    """
    Frame attribute for a string that is validated against a provided list of possible
    values for the attribute. See the `~astropy.coordinates.Attribute` API doc for
    further information.

    Parameters
    ----------
    valid_values : iterable of str
        A list or iterable of strings that define the valid values for the attribute
    default : str, None
        Default value for the attribute if not provided
    secondary_attribute : str
        Name of a secondary instance attribute which supplies the value if
        ``default is None`` and no value was supplied during initialization.
    """

    def __init__(self, valid_values, default=None, secondary_attribute=""):
        self.valid_values = list(valid_values)
        try:
            default = self.convert_input(default)[0]
        except ValueError:
            raise ValueError(
                "The specified default value is not in the list of valid values."
            )
        super().__init__(default, secondary_attribute)

    def convert_input(self, value):
        """
        Checks that the input is a valid value.

        Parameters
        ----------
        value : str
            Input value to be validated
        """

        if value is None:
            return None, False

        if value is not None and value not in self.valid_values:
            raise ValueError(
                "The specified attribute value is not in the list of valid values."
            )

        return value, False
