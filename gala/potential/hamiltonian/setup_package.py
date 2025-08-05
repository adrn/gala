from collections import defaultdict
from distutils.core import Extension


def get_extensions():
    import numpy as np

    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append(mac_incl_path)
    cfg["include_dirs"].append("gala/potential")
    cfg["extra_compile_args"].append("-std=c++17")
    cfg["sources"].append("gala/potential/hamiltonian/chamiltonian.pyx")
    cfg["sources"].append("gala/potential/hamiltonian/src/chamiltonian.cpp")
    cfg["sources"].append("gala/potential/potential/src/cpotential.cpp")
    exts.append(Extension("gala.potential.hamiltonian.chamiltonian", **cfg))

    return exts


def get_package_data():
    return {"gala.potential.hamiltonian": ["src/*.h", "src/*.cpp", "*.pyx", "*.pxd"]}
