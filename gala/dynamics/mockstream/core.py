# Third-party
import astropy.units as u
import numpy as np

# Project
from ...io import quantity_to_hdf5, quantity_from_hdf5
from .. import PhaseSpacePosition

__all__ = ['MockStream']


class MockStream(PhaseSpacePosition):

    @u.quantity_input(release_time=u.Myr)
    def __init__(self, pos, vel=None, frame=None,
                 release_time=None, lead_trail=None):

        super().__init__(pos=pos, vel=vel, frame=frame)

        if release_time is not None:
            release_time = u.Quantity(release_time)
            if len(release_time) != self.pos.shape[0]:
                raise ValueError('shape mismatch: input release time array '
                                 'must have the same shape as the input '
                                 'phase-space data, minus the component '
                                 'dimension. expected {}, got {}'
                                 .format(self.pos.shape[0],
                                         len(release_time)))

        self.release_time = release_time

        if lead_trail is not None:
            lead_trail = np.array(lead_trail)
            if len(lead_trail) != self.pos.shape[0]:
                raise ValueError('shape mismatch: input leading/trailing array '
                                 'must have the same shape as the input '
                                 'phase-space data, minus the component '
                                 'dimension. expected {}, got {}'
                                 .format(self.pos.shape[0],
                                         len(lead_trail)))

        self.lead_trail = lead_trail

    # ------------------------------------------------------------------------
    # Input / output
    #
    def to_hdf5(self, f):
        """
        Serialize this object to an HDF5 file.

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
            quantity_to_hdf5(f, 'release_time', self.release_time)

        if self.lead_trail is not None:
            f['lead_trail'] = self.lead_trail.astype('S1')  # TODO HACK
        return f

    @classmethod
    def from_hdf5(cls, f):
        """
        Load an object from an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """
        # TODO: this is duplicated code from PhaseSpacePosition
        if isinstance(f, str):
            import h5py
            f = h5py.File(f, mode='r')

        obj = PhaseSpacePosition.from_hdf5(f)

        if 'release_time' in f:
            t = quantity_from_hdf5(f['release_time'])
        else:
            t = None

        if 'lead_trail' in f:
            lt = f['lead_trail'][:]
        else:
            lt = None

        return cls(pos=obj.pos, vel=obj.vel,
                   release_time=t, lead_trail=lt,
                   frame=obj.frame)
