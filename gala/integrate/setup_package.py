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
    cfg["include_dirs"].append("gala/dynamics/nbody")
    cfg["extra_compile_args"].append("-std=c++17")
    cfg["sources"].append("gala/integrate/cyintegrators/leapfrog.pyx")
    cfg["sources"].append("gala/potential/potential/src/cpotential.cpp")
    exts.append(Extension("gala.integrate.cyintegrators.leapfrog", **cfg))

    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append(mac_incl_path)
    cfg["include_dirs"].append("gala/potential")
    cfg["extra_compile_args"].append("-std=c++17")
    cfg["sources"].append("gala/potential/hamiltonian/src/chamiltonian.cpp")
    cfg["sources"].append("gala/potential/potential/src/cpotential.cpp")
    cfg["sources"].append("gala/integrate/cyintegrators/dop853.pyx")
    cfg["sources"].append("gala/integrate/cyintegrators/dopri/dop853.cpp")
    exts.append(Extension("gala.integrate.cyintegrators.dop853", **cfg))

    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append(mac_incl_path)
    cfg["include_dirs"].append("gala/potential")
    cfg["include_dirs"].append("gala/dynamics/nbody")
    cfg["extra_compile_args"].append("-std=c++17")
    cfg["sources"].append("gala/integrate/cyintegrators/ruth4.pyx")
    cfg["sources"].append("gala/potential/potential/src/cpotential.cpp")
    exts.append(Extension("gala.integrate.cyintegrators.ruth4", **cfg))

    return exts
