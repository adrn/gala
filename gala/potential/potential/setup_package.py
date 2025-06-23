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
    cfg["include_dirs"].append("gala")
    cfg["extra_compile_args"].append("--std=gnu99")
    cfg["sources"].append("gala/potential/potential/cpotential.pyx")
    cfg["sources"].append("gala/potential/potential/builtin/builtin_potentials.c")
    cfg["sources"].append("gala/potential/potential/src/cpotential.c")
    exts.append(Extension("gala.potential.potential.cpotential", **cfg))

    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append(mac_incl_path)
    cfg["include_dirs"].append("gala/potential")
    cfg["include_dirs"].append("gala")
    cfg["extra_compile_args"].append("--std=gnu99")
    cfg["sources"].append("gala/potential/potential/ccompositepotential.pyx")
    cfg["sources"].append("gala/potential/potential/src/cpotential.c")
    exts.append(Extension("gala.potential.potential.ccompositepotential", **cfg))

    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append(mac_incl_path)
    cfg["include_dirs"].append("gala/potential")
    cfg["include_dirs"].append("gala")
    cfg["extra_compile_args"].append("--std=gnu99")
    cfg["sources"].append("gala/potential/potential/builtin/cybuiltin.pyx")
    cfg["sources"].append("gala/potential/potential/builtin/builtin_potentials.c")
    cfg["sources"].append("gala/potential/potential/builtin/multipole.c")
    cfg["sources"].append("gala/potential/potential/src/cpotential.c")
    exts.append(Extension("gala.potential.potential.builtin.cybuiltin", **cfg))

    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append(mac_incl_path)
    cfg["include_dirs"].append("gala/potential")
    cfg["include_dirs"].append("gala")
    cfg["extra_compile_args"].append("-std=c++17")
    cfg["sources"].append("gala/potential/potential/builtin/cyexp.pyx")
    cfg["sources"].append("gala/potential/potential/builtin/exp_fields.cc")
    # cfg["sources"].append("gala/potential/potential/src/cpotential.c")
    exts.append(Extension("gala.potential.potential.builtin.cyexp", **cfg))

    return exts


def get_package_data():
    return {
        "gala.potential.potential": [
            "*.h",
            "*.pyx",
            "*.pxd",
            "*/*.pyx",
            "*/*.pxd",
            "builtin/builtin_potentials.h",
            "builtin/builtin_potentials.c",
            "builtin/exp_fields.h",
            "builtin/exp_fields.cc",
            "src/cpotential.h",
            "src/cpotential.c",
            "tests/*.yml",
            "tests/pot_disk_506151.pot",
            "tests/agama_cylspline_test.fits",
        ]
    }
