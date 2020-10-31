# Third-party
import astropy.units as u


def quantity_from_hdf5(dset):
    """
    Return an Astropy Quantity object from a key in an HDF5 file,
    group, or dataset. This checks to see if the input file/group/dataset
    contains a ``'unit'`` attribute (e.g., in `f.attrs`).

    Parameters
    ----------
    dset : :class:`h5py.DataSet`

    Returns
    -------
    q : `astropy.units.Quantity`, `numpy.ndarray`
        If a unit attribute exists, this returns a Quantity. Otherwise, it
        returns a numpy array.
    """
    if 'unit' in dset.attrs and dset.attrs['unit'] is not None:
        unit = u.Unit(dset.attrs['unit'])
    else:
        unit = 1.

    return dset[:] * unit


def quantity_to_hdf5(f, key, q):
    """
    Turn an Astropy Quantity object into something we can write out to
    an HDF5 file.

    Parameters
    ----------
    f : :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
    key : str
        The name.
    q : float, `astropy.units.Quantity`
        The quantity.

    """

    if hasattr(q, 'unit'):
        f[key] = q.value
        f[key].attrs['unit'] = str(q.unit)

    else:
        f[key] = q
        f[key].attrs['unit'] = ""
