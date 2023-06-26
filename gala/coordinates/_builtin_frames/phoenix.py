"""Astropy coordinate class for the Phoenix coordinate system """

__all__ = ["PhoenixShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Phoenix",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class PhoenixShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Phoenix stream."""


R = np.array(
    [
        [0.59644670, 0.27151332, -0.75533559],
        [-0.48595429, -0.62682316, -0.60904938],
        [0.63882686, -0.73032406, 0.24192354],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, PhoenixShipp19)
def icrs_to_phoenix():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, PhoenixShipp19, ICRS)
def phoenix_to_icrs():
    return R.T
