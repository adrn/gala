"""Astropy coordinate class for the Elqui coordinate system """

__all__ = ["ElquiShipp19"]

import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Elqui",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class ElquiShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Elqui stream."""


R = np.array(
    [
        [0.74099526, 0.20483425, -0.63950681],
        [0.57756858, -0.68021616, 0.45135409],
        [0.34255009, 0.70381028, 0.62234278],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, ElquiShipp19)
def icrs_to_elqui():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, ElquiShipp19, ICRS)
def elqui_to_icrs():
    return R.T
