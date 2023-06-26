"""Astropy coordinate class for the Ravi coordinate system """

__all__ = ["RaviShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Ravi",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class RaviShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Ravi stream."""


R = np.array(
    [
        [0.57336113, -0.22475898, -0.78787081],
        [-0.57203155, -0.57862539, 0.58135407],
        [0.58654661, 0.78401279, 0.20319208],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, RaviShipp19)
def icrs_to_ravi():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, RaviShipp19, ICRS)
def ravi_to_icrs():
    return R.T
