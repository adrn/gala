"""Astropy coordinate class for the Atlas coordinate system """

__all__ = ["AtlasShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Atlas",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class AtlasShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Atlas stream."""


R = np.array(
    [
        [0.83697865, 0.29481904, -0.4610298],
        [0.51616778, -0.70514011, 0.4861566],
        [0.18176238, 0.64487142, 0.74236331],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, AtlasShipp19)
def icrs_to_atlas():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, AtlasShipp19, ICRS)
def atlas_to_icrs():
    return R.T
