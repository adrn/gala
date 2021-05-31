""" Utilities for Potential classes """

# Standard library
from functools import wraps

# Third-party
import numpy as np

# Project
from ..common import PotentialParameter
from .core import PotentialBase

__all__ = ['from_equation']
__doctest_requires__ = {('from_equation', ): ['sympy']}


def from_equation(expr, vars, pars, name=None, hessian=False):
    r"""
    Create a potential class from an expression for the potential.

    .. note::

        This utility requires having `Sympy <http://www.sympy.org/>`_ installed.

    .. warning::

        These potentials are *not* pickle-able and cannot be written
        out to YAML files (using `~gala.potential.PotentialBase.save()`)

    Parameters
    ----------
    expr : :class:`sympy.core.expr.Expr`, str
        Either a ``Sympy`` expression, or a string that can be converted to
        a ``Sympy`` expression.
    vars : iterable
        An iterable of variable names in the expression.
    pars : iterable
        An iterable of parameter names in the expression.
    name : str (optional)
        The name of the potential class returned.
    hessian : bool (optional)
        Generate a function to compute the Hessian.

    Returns
    -------
    CustomPotential : `~gala.potential.PotentialBase`
        A potential class that represents the input equation. To instantiate the
        potential, use just like a normal class with parameters.

    Examples
    --------
    Here we'll create a potential class for the harmonic oscillator
    potential, :math:`\Phi(x) = \frac{1}{2}\,k\,x^2`:

        >>> Potential = from_equation("1/2*k*x**2", vars="x", pars="k",
        ...                           name='HarmonicOscillator')
        >>> p1 = Potential(k=1.)
        >>> p1
        <HarmonicOscillatorPotential: k=1.00 (dimensionless)>

    The potential class (and object) is a fully-fledged subclass of
    `~gala.potential.PotentialBase` and therefore has many useful methods.
    For example, to integrate an orbit:

        >>> from gala.potential import Hamiltonian
        >>> H = Hamiltonian(p1)
        >>> orbit = H.integrate_orbit([1., 0], dt=0.01, n_steps=1000)

    """
    try:
        import sympy
        from sympy.utilities.lambdify import lambdify
    except ImportError:
        raise ImportError("sympy is required to use 'from_equation()' "
                          "potential class creation.")

    # convert all input to Sympy objects
    expr = sympy.sympify(expr)
    vars = [sympy.sympify(v) for v in vars]
    var_names = [v.name for v in vars]
    pars = [sympy.sympify(p) for p in pars]
    par_names = [p.name for p in pars]
    ndim = len(vars)

    # Energy / value
    energyfunc = lambdify(vars + pars, expr, dummify=False,
                          modules=['numpy', 'sympy'])

    # Gradient
    gradfuncs = []
    for var in vars:
        gradfuncs.append(lambdify(vars + pars, sympy.diff(expr, var),
                                  dummify=False,
                                  modules=['numpy', 'sympy']))

    parameters = {}
    for _name in par_names:
        parameters[_name] = PotentialParameter(_name,
                                               physical_type='dimensionless')

    class CustomPotential(PotentialBase, parameters=parameters):
        ndim = len(vars)

        def _energy(self, w, t=0.):
            kw = self.parameters.copy()
            for k, v in kw.items():
                kw[k] = v.value

            for i, name in enumerate(var_names):
                kw[name] = w[:, i]

            return np.array(energyfunc(**kw))

        def _gradient(self, w, t=0.):
            kw = self.parameters.copy()
            for k, v in kw.items():
                kw[k] = v.value

            for i, name in enumerate(var_names):
                kw[name] = w[:, i]

            grad = np.vstack([f(**kw)[np.newaxis] for f in gradfuncs])
            return grad.T

    if name is not None:
        # name = _classnamify(name)
        if "potential" not in name.lower():
            name = name + "Potential"
        CustomPotential.__name__ = str(name)

    # Hessian
    if hessian:
        hessfuncs = []
        for var1 in vars:
            for var2 in vars:
                hessfuncs.append(lambdify(vars + pars,
                                          sympy.diff(expr, var1, var2),
                                          dummify=False,
                                          modules=['numpy', 'sympy']))

        def _hessian(self, w, t):
            kw = self.parameters.copy()
            for k, v in kw.items():
                kw[k] = v.value

            for i, name in enumerate(var_names):
                kw[name] = w[:, i]

            # expand = [np.newaxis] * w[i].ndim

            # This ain't pretty, bub
            arrs = []
            for f in hessfuncs:
                hess_arr = np.array(f(**kw))
                if hess_arr.shape != w[:, i].shape:
                    hess_arr = np.tile(hess_arr, reps=w[:, i].shape)
                arrs.append(hess_arr)
            hess = np.vstack(arrs)

            return hess.reshape((ndim, ndim, len(w[:, i])))

        CustomPotential._hessian = _hessian

    CustomPotential.save = None
    return CustomPotential


def format_doc(*args, **kwargs):
    """
    Replaces the docstring of the decorated object and then formats it.

    Modeled after astropy.utils.decorators.format_doc
    """
    def set_docstring(obj):

        # None means: use the objects __doc__
        doc = obj.__doc__
        # Delete documentation in this case so we don't end up with
        # awkwardly self-inserted docs.
        obj.__doc__ = None

        # If the original has a not-empty docstring append it to the format
        # kwargs.
        kwargs['__doc__'] = obj.__doc__ or ''
        obj.__doc__ = doc.format(*args, **kwargs)
        return obj
    return set_docstring


class SympyWrapper:

    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        self = cls(**kwargs)
        if func is not None and not kwargs:
            return self(func)
        else:
            return self

    def __init__(self, func=None, var=None, include_G=True):
        if var is None:
            _var = 'x, y, z'
        else:
            _var = var
        self.var = _var
        self.include_G = include_G

    def __call__(self, wrapped_function):

        @wraps(wrapped_function)
        def wrapper(cls, *func_args, **func_kwargs):
            try:
                import sympy as sy  # noqa
            except ImportError:
                raise ImportError("Converting to a latex expression requires "
                                  "the sympy package to be installed")

            _var = sy.symbols(self.var, seq=True, real=True)
            _var = {v.name: v for v in _var}

            if cls._parameters:
                par = sy.symbols(' '.join(cls._parameters.keys()),
                                 seq=True, real=True)
                par = {v.name: v for v in par}
            else:
                par = {}

            if self.include_G:
                par['G'] = sy.symbols('G')

            return wrapped_function(cls, _var, par)

        return wrapper


sympy_wrap = SympyWrapper.as_decorator
