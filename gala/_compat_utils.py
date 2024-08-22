import numpy as np
from packaging.version import Version

# See: https://github.com/astropy/astropy/pull/16181
NUMPY_LT_2_0 = Version(np.__version__) < Version("2.0.0")
COPY_IF_NEEDED = False if NUMPY_LT_2_0 else None
