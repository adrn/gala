import astropy.units as u
import numpy as np
from scipy.spatial.transform import Rotation

from ...io import quantity_from_hdf5, quantity_to_hdf5
from .. import PhaseSpacePosition

__all__ = ["MockStream"]


class MockStream(PhaseSpacePosition):
    @u.quantity_input(release_time=u.Myr)
    def __init__(
        self, pos, vel=None, frame=None, release_time=None, lead_trail=None, copy=True
    ):
        super().__init__(pos=pos, vel=vel, frame=frame, copy=copy)

        if release_time is not None:
            release_time = u.Quantity(release_time)
            if len(release_time) != self.pos.shape[0]:
                msg = (
                    "shape mismatch: input release time array "
                    "must have the same shape as the input "
                    "phase-space data, minus the component "
                    f"dimension. expected {self.pos.shape[0]}, got {len(release_time)}"
                )
                raise ValueError(msg)

        self.release_time = release_time

        if lead_trail is not None:
            lead_trail = np.array(lead_trail)
            if len(lead_trail) != self.pos.shape[0]:
                msg = (
                    "shape mismatch: input leading/trailing array "
                    "must have the same shape as the input "
                    "phase-space data, minus the component "
                    f"dimension. expected {self.pos.shape[0]}, got {len(lead_trail)}"
                )
                raise ValueError(msg)

        self.lead_trail = lead_trail

    def rotate_to_xy_plane(self, prog_w):
        """Rotate the mock stream coordinate system to align with the x-y plane

        This method transforms the mock stream into a new coordinate system where the
        progenitor's orbital plane is aligned with the xy-plane, the stream and
        progenitor are centered at (0, 0), and the stream primarily extends in the x
        direction (leading tail at positive x and trailing tail at negative x). This is
        useful for visualizing streams in their natural orbital plane.

        Parameters
        ----------
        prog_w : `~gala.dynamics.PhaseSpacePosition`
            The phase-space position of the progenitor at the same time as the stream.
            This defines the center and orientation of the rotated coordinate system.

        Returns
        -------
        rotated_stream : `~gala.dynamics.MockStream`
            A new MockStream instance with positions and velocities transformed to the
            rotated coordinate system. The progenitor is at the origin with velocity
            aligned along the positive x-axis. The release times and lead/trail flags
            are preserved from the original stream.
        """
        R1 = Rotation.from_euler("z", -prog_w.spherical.lon.to_value(u.rad)[0])
        R2 = Rotation.from_euler("y", prog_w.spherical.lat.to_value(u.rad)[0])
        Rtmp = R2.as_matrix() @ R1.as_matrix()

        vtmp = Rtmp @ prog_w.v_xyz[:, 0]
        R3 = Rotation.from_euler("x", -np.arctan2(vtmp[2], vtmp[1]).value)
        R4 = Rotation.from_euler("z", -np.pi / 2)
        R = R4.as_matrix() @ R3.as_matrix() @ Rtmp

        prog_rot = PhaseSpacePosition(prog_w.data.transform(R))
        R_final = Rotation.from_euler(
            "z", -np.arctan2(prog_rot.v_y[0], prog_rot.v_x[0])
        ).as_matrix()

        tmp = PhaseSpacePosition(self.data.transform(R))

        return MockStream(
            pos=R_final @ (tmp.xyz - prog_rot.xyz),
            vel=R_final @ tmp.v_xyz,
            release_time=self.release_time,
            lead_trail=self.lead_trail,
        )

    # ------------------------------------------------------------------------
    # Input / output
    #
    def to_hdf5(self, f):
        """Serialize this object to an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """

        f = super().to_hdf5(f)

        # if self.potential is not None:
        #     import yaml
        #     from ..potential.potential.io import to_dict
        #     f['potential'] = yaml.dump(to_dict(self.potential)).encode('utf-8')

        if self.release_time:
            quantity_to_hdf5(f, "release_time", self.release_time)

        if self.lead_trail is not None:
            f["lead_trail"] = self.lead_trail.astype("S1")  # TODO HACK
        return f

    @classmethod
    def from_hdf5(cls, f):
        """Load an object from an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """
        # TODO: this is duplicated code from PhaseSpacePosition
        if isinstance(f, str):
            import h5py

            f = h5py.File(f, mode="r")

        obj = PhaseSpacePosition.from_hdf5(f)

        t = quantity_from_hdf5(f["release_time"]) if "release_time" in f else None

        lt = f["lead_trail"][:] if "lead_trail" in f else None

        return cls(
            pos=obj.pos, vel=obj.vel, release_time=t, lead_trail=lt, frame=obj.frame
        )
