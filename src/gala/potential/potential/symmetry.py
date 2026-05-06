"""
Symmetry classes for gravitational potentials.

These classes define coordinate transformations between symmetry-specific
coordinates (e.g., spherical radius r, cylindrical (R, z)) and the internal
Cartesian representation used by potential calculations.
"""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["CylindricalSymmetry", "PotentialSymmetry", "SphericalSymmetry"]


class PotentialSymmetry(ABC):
    """
    Base class for potential coordinate symmetries.

    This abstract base class defines the interface for converting between
    symmetry-specific coordinates and the Cartesian coordinates used internally
    by potential calculations.
    """

    @property
    @abstractmethod
    def coord_names(self):
        """
        Tuple of coordinate names for this symmetry.

        Returns
        -------
        coord_names : tuple of str
            Names of the coordinates in this symmetry system.
        """

    @abstractmethod
    def to_cartesian(self, **coords):
        """
        Convert symmetry coordinates to Cartesian coordinates.

        Parameters
        ----------
        **coords
            Coordinate values in the symmetry system. Keys must match
            the names in `coord_names`.

        Returns
        -------
        xyz : `~astropy.units.Quantity`
            Cartesian coordinates with shape (3, n_points). If inputs are
            unitless, output will also be unitless.
        """

    def validate_coords(self, **coords):
        """
        Validate that the provided coordinates are appropriate for this symmetry.

        Parameters
        ----------
        **coords
            Coordinate keyword arguments to validate.

        Raises
        ------
        ValueError
            If the coordinates are invalid or incomplete.
        """
        # Check for unexpected coordinates first - this gives a better error message
        extra = set(coords.keys()) - set(self.coord_names)
        if extra:
            raise ValueError(
                f"Invalid coordinate(s) for {self.__class__.__name__}: {extra}. "
                f"This symmetry only accepts: {self.coord_names}"
            )

        # Check that all required coordinates are provided
        # For some symmetries (like cylindrical), certain coords may be optional
        # So we only check required ones exist
        required_coords = self.coord_names  # Base class: all are required
        if hasattr(self, "_optional_coords"):
            required_coords = tuple(
                c for c in self.coord_names if c not in self._optional_coords
            )

        missing_required = set(required_coords) - set(coords.keys())
        if missing_required:
            raise ValueError(
                f"Missing required coordinate(s) for {self.__class__.__name__}: "
                f"{missing_required}. Required: {required_coords}"
            )


class SphericalSymmetry(PotentialSymmetry):
    """
    Spherical symmetry for potentials with no angular dependence.

    This symmetry is appropriate for potentials that depend only on the
    spherical radius r = sqrt(x² + y² + z²).

    Examples
    --------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> from gala.potential import HernquistPotential
    >>> pot = HernquistPotential(m=1e10*u.Msun, c=1*u.kpc)
    >>> r = np.linspace(0.1, 10, 100) * u.kpc
    >>> energy = pot.energy(r=r)
    """

    coord_names = ("r",)

    def to_cartesian(self, r):
        """
        Convert spherical radius to Cartesian coordinates.

        Parameters
        ----------
        r : array-like, `~astropy.units.Quantity`
            Spherical radius values. Can be scalar or array.

        Returns
        -------
        xyz : `~astropy.units.Quantity` or `~numpy.ndarray`
            Cartesian coordinates with shape (3, n_points). The x-component
            is set to r, while y and z are set to zero. Units are preserved
            if input has units.
        """
        # Handle units
        has_units = hasattr(r, "unit")
        if has_units:
            unit = r.unit
            r = r.value
        else:
            unit = None

        # Handle scalar vs array
        r = np.asarray(r, dtype=np.float64)
        is_scalar = r.ndim == 0
        if is_scalar:
            r = r.reshape(1)

        # Create Cartesian array: (x, y, z) = (r, 0, 0)
        xyz = np.zeros((3, r.size), dtype=np.float64)
        xyz[0] = r.ravel()

        # Reapply units if necessary
        if has_units:
            xyz = xyz * unit

        return xyz

    def validate_coords(self, **coords):
        """
        Validate spherical radius coordinate.

        Parameters
        ----------
        **coords
            Coordinate keyword arguments. Must contain 'r'.

        Raises
        ------
        ValueError
            If radius values are negative or invalid coordinates are provided.
        """
        # Call parent validation first
        super().validate_coords(**coords)

        # Now validate the value
        r = coords["r"]
        r_val = r.value if hasattr(r, "value") else r
        if np.any(r_val < 0):
            raise ValueError("Spherical radius r must be non-negative")


class CylindricalSymmetry(PotentialSymmetry):
    """
    Cylindrical (axisymmetric) symmetry for potentials with no azimuthal dependence.

    This symmetry is appropriate for potentials that depend only on the
    cylindrical radius R = sqrt(x² + y²) and height z, but not on the
    azimuthal angle phi.

    Examples
    --------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> from gala.potential import MiyamotoNagaiPotential
    >>> pot = MiyamotoNagaiPotential(m=1e11*u.Msun, a=3*u.kpc, b=0.3*u.kpc)
    >>> R = np.linspace(1, 15, 100) * u.kpc
    >>> z = np.zeros_like(R)
    >>> energy = pot.energy(R=R, z=z)
    >>>
    >>> # z can be omitted and defaults to zero
    >>> energy = pot.energy(R=R)
    """

    coord_names = ("R", "z")

    def to_cartesian(self, R, z=None):
        """
        Convert cylindrical coordinates to Cartesian coordinates.

        Parameters
        ----------
        R : array-like, `~astropy.units.Quantity`
            Cylindrical radius values. Can be scalar or array.
        z : array-like, `~astropy.units.Quantity`, optional
            Height above/below the midplane. If not provided, defaults to
            zero with the same shape as R. Must have the same shape as R
            if provided.

        Returns
        -------
        xyz : `~astropy.units.Quantity` or `~numpy.ndarray`
            Cartesian coordinates with shape (3, n_points). The x-component
            is set to R, y to 0, and z to the provided z values. Units are
            preserved if input has units.

        Raises
        ------
        ValueError
            If R and z have incompatible shapes.
        """
        # Handle units for R
        has_units = hasattr(R, "unit")
        if has_units:
            unit = R.unit
            R = R.value
        else:
            unit = None

        # Ensure array and get shape
        R = np.atleast_1d(np.asarray(R, dtype=np.float64))

        # Handle z coordinate
        if z is None:
            # Default to zeros with same shape as R
            z = np.zeros_like(R)
        else:
            # Extract units and values
            if hasattr(z, "unit"):
                if has_units and z.unit != unit:
                    # Convert z to same units as R
                    z = z.to(unit).value
                elif has_units:
                    z = z.value
                else:
                    # R has no units but z does - use z's units
                    unit = z.unit
                    z = z.value
            else:
                z = np.asarray(z, dtype=np.float64)

            z = np.atleast_1d(z)

            # Check shape compatibility
            if z.shape != R.shape:
                if z.size == 1:
                    # Broadcast scalar z to match R
                    z = np.full_like(R, z.item())
                elif R.size == 1:
                    # Broadcast scalar R to match z
                    R = np.full_like(z, R.item())
                else:
                    raise ValueError(
                        f"Incompatible shapes for R and z: R.shape={R.shape}, "
                        f"z.shape={z.shape}. Shapes must match or one must be scalar."
                    )

        # Create Cartesian array: (x, y, z) = (R, 0, z)
        xyz = np.zeros((3, R.size), dtype=np.float64)
        xyz[0] = R.ravel()
        xyz[2] = z.ravel()

        # Reapply units if necessary
        if unit is not None:
            xyz = xyz * unit

        return xyz

    def validate_coords(self, **coords):
        """
        Validate cylindrical coordinates.

        Parameters
        ----------
        **coords
            Coordinate keyword arguments. Must contain 'R', may contain 'z'.

        Raises
        ------
        ValueError
            If R values are negative or invalid coordinates are provided.
        """
        # Call parent validation first (handles checking for extra coords)
        # Mark 'z' as optional for this symmetry
        self._optional_coords = ("z",)
        super().validate_coords(**coords)

        # Now validate the values
        R = coords["R"]
        R_val = R.value if hasattr(R, "value") else R
        if np.any(R_val < 0):
            raise ValueError("Cylindrical radius R must be non-negative")
