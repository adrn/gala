# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import collections
import sys
import multiprocessing

# Third-party
from astropy import log as logger
import numpy as np

__all__ = ['get_pool', 'rolling_window', 'atleast_2d',
           'assert_angles_allclose', 'assert_quantities_allclose']

class SerialPool(object):

    def close(self):
        return

    def map(self, function, tasks, callback=None):
        results = []
        for task in tasks:
            result = function(task)
            if callback is not None:
                callback(result)
            results.append(result)
        return results

def get_pool(mpi=False, threads=None, **kwargs):
    """ Get a pool object to pass to emcee for parallel processing.
        If mpi is False and threads is None, pool is None.

        Parameters
        ----------
        mpi : bool
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
        **kwargs
            Any other keyword arguments are passed through to the pool
            initializers.
    """

    if mpi:
        from mpipool import MPIPool

        # Initialize the MPI pool
        pool = MPIPool(**kwargs)

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        logger.debug("Running with MPI...")

    elif threads is not None and threads > 1:
        logger.debug("Running with multiprocessing on {} cores..."
                     .format(threads))
        pool = multiprocessing.Pool(threads, **kwargs)

    else:
        logger.debug("Running serial...")
        pool = SerialPool(**kwargs)

    return pool

class ImmutableDict(collections.Mapping):

    @classmethod
    def from_dict(cls, somedict):
        return cls(**somedict)

    def __init__(self, **kwargs):
        self._dict = kwargs
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def __eq__(self, other):
        return self._dict == other._dict

    def __repr__(self):
        return "<ImmutableDict {}>".format(self._dict.__repr__())

    def __str__(self):
        return self._dict.__str__()

def rolling_window(arr, window_size, stride=1, return_idx=False):
    """
    There is an example of an iterator for pure-Python objects in:
    http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
    This is a rolling-window iterator Numpy arrays, with window size and
    stride control. See examples below for demos.

    Parameters
    ----------
    arr : array_like
        Input numpy array.
    window_size : int
        Width of the window.
    stride : int (optional)
        Number of indices to advance the window each iteration step.
    return_idx : bool (optional)
        Whether to return the slice indices alone with the array segment.

    Examples
    --------
    >>> a = np.array([1,2,3,4,5,6])
    >>> for x in rolling_window(a, 3):
    ...     print(x)
    [1 2 3]
    [2 3 4]
    [3 4 5]
    [4 5 6]
    >>> for x in rolling_window(a, 2, stride=2):
    ...     print(x)
    [1 2]
    [3 4]
    [5 6]
    >>> for (i1,i2),x in rolling_window(a, 2, stride=2, return_idx=True): # doctest: +SKIP
    ...     print(i1, i2, x)
    (0, 2, array([1, 2]))
    (2, 4, array([3, 4]))
    (4, 6, array([5, 6]))

    """

    window_size = int(window_size)
    stride = int(stride)

    if window_size < 0 or stride < 1:
        raise ValueError

    arr_len = len(arr)
    if arr_len < window_size:
        if return_idx:
            yield (0,arr_len),arr
        else:
            yield arr

    ix1 = 0
    while ix1 < arr_len:
        ix2 = ix1 + window_size
        result = arr[ix1:ix2]
        if return_idx:
            yield (ix1,ix2),result
        else:
            yield result
        if len(result) < window_size or ix2 >= arr_len:
            break
        ix1 += stride

def atleast_2d(*arys, **kwargs):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.
    insert_axis : int (optional)
        Where to create a new axis if input array(s) have <2 dim.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    Examples
    --------
    >>> atleast_2d(3.0)
    array([[ 3.]])

    >>> x = np.arange(3.0)
    >>> atleast_2d(x)
    array([[ 0.,  1.,  2.]])
    >>> atleast_2d(x, insert_axis=-1)
    array([[ 0.],
           [ 1.],
           [ 2.]])
    >>> atleast_2d(x).base is x
    True

    >>> atleast_2d(1, [1, 2], [[1, 2]])
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]

    """
    insert_axis = kwargs.pop('insert_axis', 0)
    slc = [slice(None)]*2
    slc[insert_axis] = None

    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1, 1)
        elif len(ary.shape) == 1:
            result = ary[slc]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

##############################################################################
# Testing
#

def assert_angles_allclose(x, y, **kwargs):
    """
    Like numpy's assert_allclose, but for angles (in radians).
    """
    c2 = (np.sin(x)-np.sin(y))**2 + (np.cos(x)-np.cos(y))**2
    diff = np.arccos((2.0 - c2)/2.0) # a = b = 1
    assert np.allclose(diff, 0.0, **kwargs)

def assert_quantities_allclose(x, y, **kwargs):
    """
    Like numpy's assert_allclose, but for quantities.
    """
    y = y.to(x.unit).value
    x = x.value
    assert np.allclose(x, y, **kwargs)
