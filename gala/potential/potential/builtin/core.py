""" Built-in potentials implemented in Cython """

# HACK: This hack brought to you by a bug in cython, and a solution from here:
# https://stackoverflow.com/questions/57138496/class-level-classmethod-can-only-be-called-on-a-method-descriptor-or-instance
try:
    myclassmethod = __builtins__.classmethod
except AttributeError:
    myclassmethod = __builtins__["classmethod"]

# Third-party
import astropy.units as u
import numpy as np
from astropy.constants import G

from gala.potential.common import PotentialParameter
from gala.potential.potential.builtin.cybuiltin import (
    BurkertWrapper,
    CylSplineWrapper,
    FlattenedNFWWrapper,
    HenonHeilesWrapper,
    HernquistWrapper,
    IsochroneWrapper,
    JaffeWrapper,
    KeplerWrapper,
    KuzminWrapper,
    LeeSutoTriaxialNFWWrapper,
    LogarithmicWrapper,
    LongMuraliBarWrapper,
    MiyamotoNagaiWrapper,
    MN3ExponentialDiskWrapper,
    MultipoleWrapper,
    NullWrapper,
    PlummerWrapper,
    PowerLawCutoffWrapper,
    SatohWrapper,
    SphericalNFWWrapper,
    StoneWrapper,
    TriaxialNFWWrapper,
)

# Project
from ..core import PotentialBase, _potential_docstring
from ..cpotential import CPotentialBase
from ..util import format_doc, sympy_wrap

__all__ = [
    "NullPotential",
    "HenonHeilesPotential",
    "KeplerPotential",
    "HernquistPotential",
    "IsochronePotential",
    "PlummerPotential",
    "JaffePotential",
    "StonePotential",
    "PowerLawCutoffPotential",
    "SatohPotential",
    "KuzminPotential",
    "MiyamotoNagaiPotential",
    "MN3ExponentialDiskPotential",
    "NFWPotential",
    "LeeSutoTriaxialNFWPotential",
    "LogarithmicPotential",
    "LongMuraliBarPotential",
    "MultipolePotential",
    "CylSplinePotential",
    "BurkertPotential",
]


def __getattr__(name):
    if name in __all__ and name in globals():
        return globals()[name]

    if not (name.startswith("MultipolePotentialLmax")):
        raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")

    if name in mp_cache:
        return mp_cache[name]

    else:
        try:
            lmax = int(name.split("Lmax")[1])
        except Exception:
            raise ImportError("Invalid")  # shouldn't ever get here!

        return make_multipole_cls(lmax, timedep="TimeDependent" in name)


@format_doc(common_doc=_potential_docstring)
class HenonHeilesPotential(CPotentialBase):
    r"""
    The Hénon-Heiles potential.

    Parameters
    ----------
    {common_doc}
    """

    ndim = 2
    Wrapper = HenonHeilesWrapper

    @myclassmethod
    @sympy_wrap(var="x y")
    def to_sympy(cls, v, p):
        expr = (
            1.0
            / 2
            * (
                v["x"] ** 2
                + v["y"] ** 2
                + 2 * v["x"] ** 2 * v["y"]
                - 2.0 / 3 * v["y"] ** 3
            )
        )
        return expr, v, p


# ============================================================================
# Spherical models
#


@format_doc(common_doc=_potential_docstring)
class KeplerPotential(CPotentialBase):
    r"""
    The Kepler potential for a point mass.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Point mass value.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    Wrapper = KeplerWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        expr = -p["G"] * p["m"] / r
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class IsochronePotential(CPotentialBase):
    r"""
    The Isochrone potential.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    b = PotentialParameter("b", physical_type="length")

    Wrapper = IsochroneWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        expr = -p["G"] * p["m"] / (sy.sqrt(r**2 + p["b"] ** 2) + p["b"])
        return expr, v, p

    def action_angle(self, w):
        """
        Transform the input cartesian position and velocity to action-angle
        coordinates the Isochrone potential. See Section 3.5.2 in
        Binney & Tremaine (2008), and be aware of the errata entry for
        Eq. 3.225.

        This transformation is analytic and can be used as a "toy potential"
        in the Sanders & Binney 2014 formalism for computing action-angle
        coordinates in _any_ potential.

        Adapted from Jason Sanders' code
        `here <https://github.com/jlsanders/genfunc>`_.

        Parameters
        ----------
        w : :class:`gala.dynamics.PhaseSpacePosition`, :class:`gala.dynamics.Orbit`
            The positions or orbit to compute the actions, angles, and
            frequencies at.
        """
        from gala.dynamics.actionangle import isochrone_xv_to_aa

        return isochrone_xv_to_aa(w, self)


@format_doc(common_doc=_potential_docstring)
class HernquistPotential(CPotentialBase):
    r"""
    Hernquist potential for a spheroid.
    See: http://adsabs.harvard.edu/abs/1990ApJ...356..359H

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    c = PotentialParameter("c", physical_type="length")

    Wrapper = HernquistWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        expr = -p["G"] * p["m"] / (r + p["c"])
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class PlummerPotential(CPotentialBase):
    r"""
    Plummer potential for a spheroid.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
       Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    b = PotentialParameter("b", physical_type="length")

    Wrapper = PlummerWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        expr = -p["G"] * p["m"] / sy.sqrt(r**2 + p["b"] ** 2)
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class JaffePotential(CPotentialBase):
    r"""
    Jaffe potential for a spheroid.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    c = PotentialParameter("c", physical_type="length")

    Wrapper = JaffeWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        expr = p["G"] * p["m"] / p["c"] * sy.log(r / (r + p["c"]))
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class StonePotential(CPotentialBase):
    r"""
    StonePotential(m, r_c, r_h, units=None, origin=None, R=None)

    Stone potential from `Stone & Ostriker (2015)
    <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    Parameters
    ----------
    m_tot : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Core radius.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
        Halo radius.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    r_c = PotentialParameter("r_c", physical_type="length")
    r_h = PotentialParameter("r_h", physical_type="length")

    Wrapper = StoneWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        A = -2 * p["G"] * p["m"] / (np.pi * (p["r_h"] - p["r_c"]))
        expr = A * (
            p["r_h"] / r * sy.atan(r / p["r_h"])
            - p["r_c"] / r * sy.atan(r / p["r_c"])
            + 1.0 / 2 * sy.log((r**2 + p["r_h"] ** 2) / (r**2 + p["r_c"] ** 2))
        )
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class PowerLawCutoffPotential(CPotentialBase, GSL_only=True):
    r"""
    A spherical power-law density profile with an exponential cutoff.

    The power law index must be ``0 <= alpha < 3``.

    .. note::

        This potential requires GSL to be installed, and Gala must have been
        built and installed with GSL support enaled (the default behavior).
        See http://gala.adrian.pw/en/latest/install.html for more information.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    alpha : numeric
        Power law index. Must satisfy: ``alpha < 3``
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Cutoff radius.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    alpha = PotentialParameter("alpha", physical_type="dimensionless")
    r_c = PotentialParameter("r_c", physical_type="length")

    Wrapper = PowerLawCutoffWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        G = p["G"]
        m = p["m"]
        alpha = p["alpha"]
        r_c = p["r_c"]
        r = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)

        expr = (
            G
            * alpha
            * m
            * sy.lowergamma(3.0 / 2 - alpha / 2, r**2 / r_c**2)
            / (2 * r * sy.gamma(5.0 / 2 - alpha / 2))
            + G
            * m
            * sy.lowergamma(1 - alpha / 2, r**2 / r_c**2)
            / (r_c * sy.gamma(3.0 / 2 - alpha / 2))
            - 3
            * G
            * m
            * sy.lowergamma(3.0 / 2 - alpha / 2, r**2 / r_c**2)
            / (2 * r * sy.gamma(5.0 / 2 - alpha / 2))
        )

        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class BurkertPotential(CPotentialBase):
    r"""
    The Burkert potential that well-matches the rotation curve of dwarf galaxies.
    See https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html

    Parameters
    ----------
    rho : :class:`~astropy.units.Quantity`, numeric [mass density]
        Central mass density.
    r0 : :class:`~astropy.units.Quantity`, numeric [length]
        The core radius.
    {common_doc}
    """

    rho = PotentialParameter("rho", physical_type="mass density")
    r0 = PotentialParameter("r0", physical_type="length")

    Wrapper = BurkertWrapper

    
    @classmethod
    def from_r0(cls, r0, units=None):
        r"""
        from_r0(r0, units=None)

        Initialize a Burkert potential from the core radius, ``r0``.
        See Equations 4 and 5 of Mori & Burkert.

        Parameters
        ----------
        r0 : :class:`~astropy.units.Quantity`, numeric [length]
            The core radius of the Burkert potential.
        """
        a = 0.021572405792749372 * u.Msun / u.pc**3  # converted: 1.46e-24 g/cm**3
        rho_d0 = a * (r0 / (3.07 * u.kpc))**(-2/3)
        return cls(rho=rho_d0, r0=r0, units=units)


# ============================================================================
# Flattened, axisymmetric models
#


@format_doc(common_doc=_potential_docstring)
class SatohPotential(CPotentialBase):
    r"""
    SatohPotential(m, a, b, units=None, origin=None, R=None)

    Satoh potential for a flattened mass distribution.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scale height.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    a = PotentialParameter("a", physical_type="length")
    b = PotentialParameter("b", physical_type="length")

    Wrapper = SatohWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        R = sy.sqrt(v["x"] ** 2 + v["y"] ** 2)
        z = v["z"]
        term = R**2 + z**2 + p["a"] * (p["a"] + 2 * sy.sqrt(z**2 + p["b"] ** 2))
        expr = -p["G"] * p["m"] / sy.sqrt(term)
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class KuzminPotential(CPotentialBase):
    r"""
    KuzminPotential(m, a, units=None, origin=None, R=None)

    Kuzmin potential for a flattened mass distribution.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Flattening parameter.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    a = PotentialParameter("a", physical_type="length")

    Wrapper = KuzminWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        denom = sy.sqrt(v["x"] ** 2 + v["y"] ** 2 + (p["a"] + sy.Abs(v["z"])) ** 2)
        expr = -p["G"] * p["m"] / denom
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class MiyamotoNagaiPotential(CPotentialBase):
    r"""
    MiyamotoNagaiPotential(m, a, b, units=None, origin=None, R=None)

    Miyamoto-Nagai potential for a flattened mass distribution.

    See: http://adsabs.harvard.edu/abs/1975PASJ...27..533M

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scale height.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    a = PotentialParameter("a", physical_type="length")
    b = PotentialParameter("b", physical_type="length")

    Wrapper = MiyamotoNagaiWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        R = sy.sqrt(v["x"] ** 2 + v["y"] ** 2)
        z = v["z"]
        term = R**2 + (p["a"] + sy.sqrt(z**2 + p["b"] ** 2)) ** 2
        expr = -p["G"] * p["m"] / sy.sqrt(term)
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class MN3ExponentialDiskPotential(CPotentialBase):
    """
    MN3ExponentialDiskPotential(m, h_R, h_z, positive_density=True, sech2_z=True,
    units=None, origin=None, R=None)

    A sum of three Miyamoto-Nagai disk potentials that approximate the potential
    generated by a double exponential disk.

    This model is taken from `Smith et al. (2015)
    <https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract>`_ - if you
    use this potential class, please also cite that work.

    As described in the above reference, this approximation has two options: (1)
    with the ``positive_density=True`` argument set, this density will be
    positive everywhere, but is only a good approximation of the exponential
    density within about 5 disk scale lengths, and (2) with
    ``positive_density=False``, this density will be negative in some regions,
    but is a better approximation out to about 7 or 8 disk scale lengths.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    h_R : :class:`~astropy.units.Quantity`, numeric [length]
        Radial (exponential) scale length.
    h_z : :class:`~astropy.units.Quantity`, numeric [length]
        If ``sech2_z=True``, this is the scale height in a sech^2 vertical
        profile. If ``sech2_z=False``, this is an exponential scale height.
    {common_doc}

    """

    m = PotentialParameter("m", physical_type="mass")
    h_R = PotentialParameter("h_R", physical_type="length")
    h_z = PotentialParameter("h_z", physical_type="length")
    Wrapper = MN3ExponentialDiskWrapper

    _K_pos_dens = np.array(
        [
            [0.0036, -0.0330, 0.1117, -0.1335, 0.1749],
            [-0.0131, 0.1090, -0.3035, 0.2921, -5.7976],
            [-0.0048, 0.0454, -0.1425, 0.1012, 6.7120],
            [-0.0158, 0.0993, -0.2070, -0.7089, 0.6445],
            [-0.0319, 0.1514, -0.1279, -0.9325, 2.6836],
            [-0.0326, 0.1816, -0.2943, -0.6329, 2.3193],
        ]
    )
    _K_neg_dens = np.array(
        [
            [-0.0090, 0.0640, -0.1653, 0.1164, 1.9487],
            [0.0173, -0.0903, 0.0877, 0.2029, -1.3077],
            [-0.0051, 0.0287, -0.0361, -0.0544, 0.2242],
            [-0.0358, 0.2610, -0.6987, -0.1193, 2.0074],
            [-0.0830, 0.4992, -0.7967, -1.2966, 4.4441],
            [-0.0247, 0.1718, -0.4124, -0.5944, 0.7333],
        ]
    )

    def __init__(
        self,
        *args,
        units=None,
        origin=None,
        R=None,
        positive_density=True,
        sech2_z=True,
        **kwargs,
    ):
        PotentialBase.__init__(self, *args, units=units, origin=origin, R=R, **kwargs)
        hzR = (self.parameters["h_z"] / self.parameters["h_R"]).decompose()

        if positive_density:
            K = self._K_pos_dens
        else:
            K = self._K_neg_dens

        # get b / h_R
        if sech2_z:
            b_hR = -0.033 * hzR**3 + 0.262 * hzR**2 + 0.659 * hzR
        else:
            b_hR = -0.269 * hzR**3 + 1.08 * hzR**2 + 1.092 * hzR

        self.positive_density = positive_density
        self.sech2_z = sech2_z

        x = np.vander([b_hR], N=5)[0]

        param_vec = K @ x

        self._ms = param_vec[:3] * self.parameters["m"].value
        self._as = param_vec[3:] * self.parameters["h_R"].value
        self._b = b_hR * self.parameters["h_R"]

        c_only = {}
        for i in range(3):
            c_only[f"m{i+1}"] = self._ms[i]
            c_only[f"a{i+1}"] = self._as[i]
            c_only[f"b{i+1}"] = self._b.value

        self._setup_wrapper(c_only)

    def get_three_potentials(self):
        """
        Return three MiyamotoNagaiPotential instances that represent the three internal
        components of this MN3 potential model
        """
        pots = {}
        for i in range(3):
            name = f"disk{i+1}"
            pots[name] = MiyamotoNagaiPotential(
                m=self._ms[i], a=self._as[i], b=self._b, units=self.units
            )
        return pots


# ============================================================================
# Triaxial models
#


@format_doc(common_doc=_potential_docstring)
class NFWPotential(CPotentialBase):
    r"""
    NFWPotential(m, r_s, a=1, b=1, c=1, units=None, origin=None, R=None)

    General Navarro-Frenk-White potential. Supports spherical, flattened, and
    triaxiality but the flattening is introduced into the potential, not the
    density, and can therefore lead to unphysical mass distributions. For a
    triaxial NFW potential that supports flattening in the density, see
    :class:`gala.potential.LeeSutoTriaxialNFWPotential`.

    See also the alternate initializers: `NFWPotential.from_circular_velocity`
    and `NFWPotential.from_M200_c`

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Scale mass.
    r_s : :class:`~astropy.units.Quantity`, numeric [length]
        Scale radius.
    a : numeric
        Major axis scale.
    b : numeric
        Intermediate axis scale.
    c : numeric
        Minor axis scale.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    r_s = PotentialParameter("r_s", physical_type="length")
    a = PotentialParameter("a", physical_type="dimensionless", default=1.0)
    b = PotentialParameter("b", physical_type="dimensionless", default=1.0)
    c = PotentialParameter("c", physical_type="dimensionless", default=1.0)

    def _setup_potential(self, parameters, origin=None, R=None, units=None):
        super()._setup_potential(parameters, origin=origin, R=R, units=units)

        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        if np.allclose([a, b, c], 1.0):
            self.Wrapper = SphericalNFWWrapper

        elif np.allclose([a, b], 1.0):
            self.Wrapper = FlattenedNFWWrapper

        else:
            self.Wrapper = TriaxialNFWWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        uu = (
            sy.sqrt(
                (v["x"] / p["a"]) ** 2 + (v["y"] / p["b"]) ** 2 + (v["z"] / p["c"]) ** 2
            )
            / p["r_s"]
        )
        v_h2 = p["G"] * p["m"] / p["r_s"]
        expr = -v_h2 * sy.log(1 + uu) / uu
        return expr, v, p

    @staticmethod
    def from_M200_c(M200, c, rho_c=None, units=None, origin=None, R=None):
        r"""
        from_M200_c(M200, c, rho_c=None, units=None, origin=None, R=None)

        Initialize an NFW potential from a virial mass, ``M200``, and a
        concentration, ``c``.

        Parameters
        ----------
        M200 : :class:`~astropy.units.Quantity`, numeric [mass]
            Virial mass, or mass at 200 times the critical density, ``rho_c``.
        c : numeric
            NFW halo concentration.
        rho_c : :class:`~astropy.units.Quantity`, numeric [density]
            Critical density at z=0. If not specified, uses the default astropy
            cosmology to obtain this, `~astropy.cosmology.default_cosmology`.
        """
        if rho_c is None:
            from astropy.constants import G
            from astropy.cosmology import default_cosmology

            cosmo = default_cosmology.get()
            rho_c = (3 * cosmo.H(0.0) ** 2 / (8 * np.pi * G)).to(u.Msun / u.kpc**3)

        Rvir = np.cbrt(M200 / (200 * rho_c) / (4.0 / 3 * np.pi)).to(u.kpc)
        r_s = Rvir / c

        A_NFW = np.log(1 + c) - c / (1 + c)
        m = M200 / A_NFW

        return NFWPotential(
            m=m, r_s=r_s, a=1.0, b=1.0, c=1.0, units=units, origin=origin, R=R
        )

    @staticmethod
    def from_circular_velocity(
        v_c,
        r_s,
        a=1.0,
        b=1.0,
        c=1.0,
        r_ref=None,
        units=None,
        origin=None,
        R=None,
    ):
        r"""
        Initialize an NFW potential from a circular velocity, scale radius, and
        reference radius for the circular velocity.

        For scale mass :math:`m_s`, scale radius :math:`r_s`, scaled
        reference radius :math:`u_{\rm ref} = r_{\rm ref}/r_s`:

        .. math::

            \frac{G\,m_s}{r_s} = \frac{v_c^2}{u_{\rm ref}} \,
                \left[\frac{u_{\rm ref}}{1+u_{\rm ref}} -
                \frac{\ln(1+u_{\rm ref})}{u_{\rm ref}^2} \right]^{-1}

        Parameters
        ----------
        v_c : :class:`~astropy.units.Quantity`, numeric [velocity]
            Circular velocity at the reference radius ``r_ref`` (see below).
        r_s : :class:`~astropy.units.Quantity`, numeric [length]
            Scale radius.
        a : numeric
            Major axis scale.
        b : numeric
            Intermediate axis scale.
        c : numeric
            Minor axis scale.
        r_ref : :class:`~astropy.units.Quantity`, numeric [length] (optional)
            Reference radius at which the circular velocity is given. By default,
            this is assumed to be the scale radius, ``r_s``.

        """

        if not hasattr(v_c, "unit"):
            v_c = v_c * units["length"] / units["time"]

        if not hasattr(r_s, "unit"):
            r_s = r_s * units["length"]

        if r_ref is None:
            r_ref = r_s

        m = NFWPotential._vc_rs_rref_to_m(v_c, r_s, r_ref)
        m = m.to(units["mass"])

        return NFWPotential(
            m=m, r_s=r_s, a=a, b=b, c=c, units=units, origin=origin, R=R
        )

    @staticmethod
    def _vc_rs_rref_to_m(v_c, r_s, r_ref):
        uu = r_ref / r_s
        vs2 = v_c**2 / uu / (np.log(1 + uu) / uu**2 - 1 / (uu * (1 + uu)))
        return r_s * vs2 / G


@format_doc(common_doc=_potential_docstring)
class LogarithmicPotential(CPotentialBase):
    r"""
    LogarithmicPotential(v_c, r_h, q1, q2, q3, phi=0, theta=0, psi=0, units=None,
    origin=None, R=None)

    Triaxial logarithmic potential.

    Parameters
    ----------
    v_c : :class:`~astropy.units.Quantity`, numeric [velocity]
        Circular velocity.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
        Scale radius.
    q1 : numeric
        Flattening in X.
    q2 : numeric
        Flattening in Y.
    q3 : numeric
        Flattening in Z.
    phi : `~astropy.units.Quantity`, numeric
        First euler angle in the z-x-z convention.
    {common_doc}
    """

    v_c = PotentialParameter("v_c", physical_type="speed")
    r_h = PotentialParameter("r_h", physical_type="length")
    q1 = PotentialParameter("q1", physical_type="dimensionless", default=1.0)
    q2 = PotentialParameter("q2", physical_type="dimensionless", default=1.0)
    q3 = PotentialParameter("q3", physical_type="dimensionless", default=1.0)
    phi = PotentialParameter("phi", physical_type="angle", default=0.0)

    Wrapper = LogarithmicWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        r2 = (v["x"] / p["q1"]) ** 2 + (v["y"] / p["q2"]) ** 2 + (v["z"] / p["q3"]) ** 2
        expr = 1.0 / 2 * p["v_c"] ** 2 * sy.log(p["r_h"] ** 2 + r2)
        return expr, v, p


@format_doc(common_doc=_potential_docstring)
class LeeSutoTriaxialNFWPotential(CPotentialBase):
    r"""
    LeeSutoTriaxialNFWPotential(v_c, r_s, a, b, c, units=None, origin=None, R=None)

    Approximation of a Triaxial NFW Potential with the flattening in the density,
    not the potential.
    See `Lee & Suto (2003) <http://adsabs.harvard.edu/abs/2003ApJ...585..151L>`_
    for details.

    Parameters
    ----------
    v_c : `~astropy.units.Quantity`, numeric [velocity]
        Circular velocity at the scale radius.
    r_h : `~astropy.units.Quantity`, numeric [length]
        Scale radius.
    a : numeric
        Major axis.
    b : numeric
        Intermediate axis.
    c : numeric
        Minor axis.
    {common_doc}
    """

    v_c = PotentialParameter("v_c", physical_type="speed")
    r_s = PotentialParameter("r_s", physical_type="length")
    a = PotentialParameter("a", physical_type="dimensionless", default=1.0)
    b = PotentialParameter("b", physical_type="dimensionless", default=1.0)
    c = PotentialParameter("c", physical_type="dimensionless", default=1.0)

    Wrapper = LeeSutoTriaxialNFWWrapper

    # TODO: implement to_sympy()


@format_doc(common_doc=_potential_docstring)
class LongMuraliBarPotential(CPotentialBase):
    r"""
    LongMuraliBarPotential(m, a, b, c, alpha=0, units=None, origin=None, R=None)

    A simple, triaxial model for a galaxy bar. This is a softened "needle"
    density distribution with an analytic potential form.
    See `Long & Murali (1992) <http://adsabs.harvard.edu/abs/1992ApJ...397...44L>`_
    for details.

    Parameters
    ----------
    m : `~astropy.units.Quantity`, numeric [mass]
        Mass scale.
    a : `~astropy.units.Quantity`, numeric [length]
        Bar half-length.
    b : `~astropy.units.Quantity`, numeric [length]
        Like the Miyamoto-Nagai ``b`` parameter.
    c : `~astropy.units.Quantity`, numeric [length]
        Like the Miyamoto-Nagai ``c`` parameter.
    {common_doc}
    """

    m = PotentialParameter("m", physical_type="mass")
    a = PotentialParameter("a", physical_type="length")
    b = PotentialParameter("b", physical_type="length")
    c = PotentialParameter("c", physical_type="length")
    alpha = PotentialParameter("alpha", physical_type="angle", default=0)

    Wrapper = LongMuraliBarWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        x = v["x"] * sy.cos(p["alpha"]) + v["y"] * sy.sin(p["alpha"])
        y = -v["x"] * sy.sin(p["alpha"]) + v["y"] * sy.cos(p["alpha"])
        z = v["z"]

        Tm = sy.sqrt(
            (p["a"] - x) ** 2 + y**2 + (p["b"] + sy.sqrt(p["c"] ** 2 + z**2)) ** 2
        )
        Tp = sy.sqrt(
            (p["a"] + x) ** 2 + y**2 + (p["b"] + sy.sqrt(p["c"] ** 2 + z**2)) ** 2
        )

        expr = (
            p["G"]
            * p["m"]
            / (2 * p["a"])
            * sy.log((x - p["a"] + Tm) / (x + p["a"] + Tp))
        )

        return expr, v, p


# ==============================================================================
# Special
#


@format_doc(common_doc=_potential_docstring)
class NullPotential(CPotentialBase):
    r"""
    NullPotential(units=None, origin=None, R=None)

    A null potential with 0 mass. Does nothing.

    Parameters
    ----------
    {common_doc}
    """

    Wrapper = NullWrapper


# ==============================================================================
# Multipole and flexible potential models
#
mp_cache = {}


def make_multipole_cls(lmax, timedep=False):
    """Create a MultipolePotential or MultipoleTimeDependentPotential class
    (not an instance!) with the specified value of lmax.

    Parameters:
    -----------
    lmax : int
    timedep : bool

    """
    if timedep:
        raise NotImplementedError("Time dependent potential coming soon!")
        # cls = MultipoleTimeDependentPotential
        # param_default = [0.]
    else:
        cls = MultipolePotential
        param_default = 0.0
    cls_name = f"{cls.__name__}Lmax{lmax}"

    if cls_name in mp_cache:
        return mp_cache[cls_name]

    parameters = {
        "_lmax": lmax,
        "inner": PotentialParameter("inner", default=False),
        "m": PotentialParameter("m", physical_type="mass", default=1.0),
        "r_s": PotentialParameter("r_s", physical_type="length", default=1.0),
    }
    doc_lines = []
    ab_callsig = []
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            if timedep:
                a = f"alpha{l}{m}"
                b = f"beta{l}{m}"
                dtype = "array-like"
            else:
                a = f"S{l}{m}"
                b = f"T{l}{m}"
                dtype = "float"

            parameters[a] = PotentialParameter(
                a, physical_type="dimensionless", default=param_default
            )
            parameters[b] = PotentialParameter(
                b, physical_type="dimensionless", default=param_default
            )

            doc_lines.append(f"{a} : {dtype}\n{b} : {dtype}")
            ab_callsig.append(f"{a}, {b}")

    ab_callsig = ", ".join(ab_callsig)
    call_signature = f"{cls.__name__}(m, r_s, {ab_callsig})"
    parameters["__doc__"] = call_signature + cls.__doc__ + "\n".join(doc_lines)

    # https://stackoverflow.com/a/58716798/623453
    parameters["__module__"] = __name__

    # Create a new SkyOffsetFrame subclass for this frame class.
    potential_cls = type(cls_name, (cls,), parameters)
    mp_cache[cls_name] = potential_cls
    return mp_cache[cls_name]


class MultipolePotential(CPotentialBase, GSL_only=True):
    r"""

    A perturbing potential represented by a multipole expansion.

    Inner:

    .. math::

        \Phi^l_\mathrm{max}(r,\theta,\phi) = \sum_{l=1}^{l=l_\mathrm{max}}\sum_{m=0}^{m=l}
            r^l \, (S_{lm} \, \cos{m\,\phi} + T_{lm} \, \sin{m\,\phi})
            \, P_l^m(\cos\theta)

    Outer:

    .. math::

        \Phi^l_\mathrm{max}(r,\theta,\phi) = \sum_{l=1}^{l=l_\mathrm{max}}\sum_{m=0}^{m=l}
            r^{-(l+1)} \, (S_{lm} \, \cos{m\,\phi} + T_{lm} \, \sin{m\,\phi})
            \, P_l^m(\cos\theta)


    The allowed coefficient parameter names will depend on how you set ``lmax``, and the
    default value for all coefficient parameter values is 0.

    Parameters
    ----------
    m : numeric
        Scale mass.
    r_s : numeric
        Scale length.
    lmax : int
        The maximum ``l`` order.
    inner : bool (optional)
        Controls whether to use the inner expansion, or the outer expansion (see above).
        Default value = ``False``.
    S00 : float (optional)
    S10 : float (optional)
    S11 : float (optional)
    T11 : float (optional)
    etc.

    Examples
    --------
    To create a potential object with only a dipole:

        >>> pot = MultipolePotential(lmax=1, S10=5.)
    """

    Wrapper = MultipoleWrapper

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        kwargs.pop("lmax", None)

        PotentialBase.__init__(self, *args, units=units, origin=origin, R=R, **kwargs)

        self._setup_wrapper(
            {"lmax": self._lmax, "n_coeffs": sum(range(self._lmax + 2))}
        )

    def __new__(cls, *args, **kwargs):
        # We don't want to call this method if we've already set up
        # an skyoffset frame for this class.
        if not (issubclass(cls, MultipolePotential) and cls is not MultipolePotential):
            try:
                lmax = kwargs["lmax"]
            except KeyError:
                raise TypeError(
                    "Can't initialize a MultipolePotential without specifying "
                    "the `lmax` keyword argument."
                )
            newcls = make_multipole_cls(lmax)
            return newcls.__new__(newcls, *args, **kwargs)

        if super().__new__ is object.__new__:
            return super().__new__(cls)
        return super().__new__(cls, *args, **kwargs)


@format_doc(common_doc=_potential_docstring)
class CylSplinePotential(CPotentialBase):
    r"""
    A flexible potential model that uses spline interpolation over a 2D grid in
    cylindrical R-z coordinates.

    Parameters
    ----------
    grid_R : `~astropy.units.Quantity`, numeric [length]
        A 1D grid of cylindrical radius R values. This should start at 0.
    grid_z : `~astropy.units.Quantity`, numeric [length]
        A 1D grid of cylindrical z values. This should start at 0.
    grid_Phi : `~astropy.units.Quantity`, numeric [specific energy]
        A 2D grid of potential values, evaluated at all R,z locations.
    {common_doc}
    """

    grid_R = PotentialParameter("grid_R", physical_type="length")
    grid_z = PotentialParameter("grid_z", physical_type="length")
    grid_Phi = PotentialParameter("grid_Phi", physical_type="specific energy")

    Wrapper = CylSplineWrapper

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Load a potential instance from an Agama export file.

        Parameters
        ----------
        filename : path-like
            The path to the Agama expoirt file, either as a string or ``pathlib.Path`` object.
        **kwargs
            Other keyword arguments are passed to the initializer.
        """
        with open(filename, "r") as f:
            raw_lines = f.readlines()

        start = r"#R(row)\z(col)"
        Phi_lines = []
        for i, line in enumerate(raw_lines):
            if line.startswith(start):
                Phi_lines.append(
                    [np.nan]
                    + [float(y) for y in line[len(start) :].strip().split("\t")]
                )
                break

        Phi_lines.extend(
            [[float(y) for y in x.strip().split("\t")] for x in raw_lines[i + 1 :]]
        )
        Phi_lines = np.array(Phi_lines)

        gridR = Phi_lines[1:, 0] * u.kpc
        gridz = Phi_lines[0, 1:] * u.kpc
        gridPhi = Phi_lines[1:, 1:] * (u.km / u.s) ** 2

        return cls(gridR, gridz, gridPhi, **kwargs)

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        PotentialBase.__init__(self, *args, units=units, origin=origin, R=R, **kwargs)

        grid_R = self.parameters["grid_R"]
        grid_z = self.parameters["grid_z"]
        grid_Phi = self.parameters["grid_Phi"]
        Phi0 = grid_Phi[0, 0]  # potential at R=0,z=0

        self._multipole_pot = self._fit_asympt(grid_R, grid_z, grid_Phi)
        Phi_Rmax = self._multipole_pot.energy([1.0, 0, 0] * grid_R.max())
        Mtot = -Phi_Rmax[0] * grid_R.max()

        if Phi0 < 0 and Mtot > 0:
            # assign Rscale so that it approximately equals -Mtotal/Phi(r=0),
            # i.e. would equal the scale radius of a Plummer potential
            Rscale = (-Mtot / Phi0).to(self.units["length"])
        else:
            Rscale = grid_R[len(grid_R) // 2]  # "rather arbitrary"

        # APW: assumed / enforced mmax=0 - different from Agama

        sizeR = len(grid_R)

        # grid in z assumed to only cover half-space z>=0; the density is assumed
        # to be z-reflection symmetric:
        sizez_orig = len(grid_z)
        grid_z = np.concatenate((-grid_z[::-1], grid_z[1:]))
        sizez = len(grid_z)

        # transform the grid to log-scaled coordinates
        grid_R_asinh = np.arcsinh((grid_R / Rscale).decompose().value)
        grid_z_asinh = np.arcsinh((grid_z / Rscale).decompose().value)

        logScaling = np.all(grid_Phi < 0)

        # temporary containers of scaled potential and derivatives used to
        # construct 2d splines

        if grid_Phi.shape[0] != sizeR or grid_Phi.shape[1] != sizez_orig:
            raise ValueError("CylSpline: incorrect coefs array size")

        grid_Phi_full = np.zeros((sizeR, sizez))
        grid_Phi_full[:, : sizez_orig - 1] = grid_Phi[:, :0:-1]
        grid_Phi_full[:, sizez_orig - 1 :] = grid_Phi
        if logScaling:
            grid_Phi_full = np.log(-grid_Phi_full)
        else:
            grid_Phi_full = grid_Phi_full

        from scipy.interpolate import RectBivariateSpline

        self.spl = RectBivariateSpline(grid_R_asinh, grid_z_asinh, grid_Phi_full)

        # Note: if MultipolePotential parameter order changes, this needs to be updated!
        multipole_pars = np.concatenate(
            [
                [
                    self.G,
                    self._multipole_pot._lmax,
                    sum(range(self._multipole_pot._lmax + 2)),
                ],
                [x.value for x in self._multipole_pot.parameters.values()],
            ]
        )

        self._c_only = {
            "log_scaling": logScaling,
            "Rscale": Rscale.value,
            "sizeR": sizeR,
            "sizez": sizez,
            "grid_R_trans": grid_R_asinh,
            "grid_z_trans": grid_z_asinh,
            "grid_Phi_trans": grid_Phi_full.T,
            "multipole_pars": multipole_pars,
        }
        self._setup_wrapper(self._c_only)

    def _fit_asympt(self, grid_R, grid_z, grid_Phi, lmax_fit=8):
        """
        Assumes z reflection symmetry

        lmax_fit : int
            Number of meridional harmonics to fit - don't set too large

        """
        from scipy.special import sph_harm

        sizeR = len(grid_R)
        sizez = len(grid_z)

        # assemble the boundary points and their indices
        assert grid_Phi.shape == (sizeR, sizez)
        maxz = np.max(grid_z.value)

        # first run along R at the max-z and min-z edges
        points = np.concatenate(
            ([[R, maxz] for R in grid_R.value], [[R, -maxz] for R in grid_R.value])
        )
        Phis = np.concatenate(
            (grid_Phi[:, np.argmax(grid_z)].value, grid_Phi[:, np.argmax(grid_z)].value)
        )

        maxR = np.max(grid_R.value)
        points = np.concatenate(
            (
                points,
                [[maxR, z] for z in grid_z.value],
                [[maxR, -z] for z in grid_z.value],
            )
        )
        Phis = np.concatenate(
            (
                Phis,
                grid_Phi[np.argmax(grid_R), :].value,
                grid_Phi[np.argmax(grid_R), :].value,
            )
        )

        npoints = len(points)
        # ncoefs = lmax_fit + 1

        r0 = min(np.max(grid_R), np.max(grid_z))

        i, j = len(grid_R) // 2, len(grid_z) // 2
        rr = np.sqrt(grid_R[i] ** 2 + grid_z[j] ** 2)
        m = np.abs(grid_Phi[i, j] * rr / G).to(self.units["mass"])
        scale = (G * m / r0).decompose(self.units).value

        # find values of spherical harmonic coefficients
        # that best match the potential at the array of boundary points

        # for m-th harmonic, we may have lmax-m+1 different l-terms
        matr = np.zeros((npoints, lmax_fit + 1))

        # The linear system to solve in the least-square sense is M_{p,l} * S_l = R_p,
        # where R_p = Phi at p-th boundary point (0<=p<npoints),
        # M_{l,p}   = value of l-th harmonic coefficient at p-th boundary point,
        # S_l       = the amplitude of l-th coefficient to be determined.
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        theta = np.arctan2(points[:, 0], points[:, 1])

        ls = np.arange(lmax_fit + 1)
        Pl0 = np.stack([sph_harm(0, l, 0.0, theta).real for l in ls]).T

        matr = (r[:, None] / r0.value) ** -(ls[None] + 1) * Pl0
        y = Phis / scale
        sol, resid, rank, s = np.linalg.lstsq(matr, y, rcond=None)

        pars = {f"S{l}0": sol[l].real for l in ls}
        return MultipolePotential(
            lmax=lmax_fit, m=m, r_s=r0, inner=False, units=self.units, **pars
        )
