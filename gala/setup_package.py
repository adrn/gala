from distutils.core import Extension
from collections import defaultdict


def get_extensions():
    exts = []

    cfg = defaultdict(list)
    cfg["include_dirs"].append("gala")
    cfg["sources"].append("gala/cconfig.pyx")
    exts.append(Extension("gala._cconfig", **cfg))

    return exts
