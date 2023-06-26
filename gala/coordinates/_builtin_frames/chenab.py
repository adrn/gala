"""Astropy coordinate class for the Chenab coordinate system """

__all__ = ["ChenabShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Chenab",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class ChenabShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Chenab stream."""


R = np.array(
    [
        [0.51883185, -0.34132444, -0.78378003],
        [-0.81981696, 0.06121342, -0.56934442],
        [-0.24230902, -0.93795018, 0.24806410],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, ChenabShipp19)
def icrs_to_chenab():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, ChenabShipp19, ICRS)
def chenab_to_icrs():
    return R.T
