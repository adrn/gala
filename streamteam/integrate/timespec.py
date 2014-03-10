# coding: utf-8

""" Helper function for turning different ways of specifying the integration
    times into an array of times.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

def _parse_time_specification(dt=None, nsteps=None, t1=None, t2=None, t=None):
    """ Return an array of times given a few combinations of kwargs that are
        accepted -- see below.

        Parameters
        ----------
        dt, nsteps[, t1] : (numeric, int[, numeric])
            A fixed timestep dt and a number of steps to run for.
        dt, t1, t2 : (numeric, numeric, numeric)
            A fixed timestep dt, an initial time, and an final time.
        dt, t1 : (array_like, numeric)
            An array of timesteps dt and an initial time.
        nsteps, t1, t2 : (int, numeric, numeric)
            Number of steps between an initial time, and a final time.
        t : array_like
            An array of times (dts = t[1:] - t[:-1])

    """

    # t : array_like
    if t is not None:
        times = t
        return times

    else:
        if dt is None and (t1 is None or t2 is None or nsteps is None):
            raise ValueError("Invalid spec. See docstring.")

        # dt, nsteps[, t1] : (numeric, int[, numeric])
        elif dt is not None and nsteps is not None:
            if t1 is None:
                t1 = 0.

            times = _parse_time_specification(dt=np.ones(nsteps)*dt,
                                              t1=t1)
        # dt, t1, t2 : (numeric, numeric, numeric)
        elif dt is not None and t1 is not None and t2 is not None:
            if t2 < t1 and dt < 0:

                t_i = t1
                times = []
                ii = 0
                while (t_i > t2) and (ii < 1E6):
                    times.append(t_i)
                    t_i += dt

                if times[-1] != t2:
                    times.append(t2)

                return np.array(times)

            elif t2 > t1 and dt > 0:

                t_i = t1
                times = []
                ii = 0
                while (t_i < t2) and (ii < 1E6):
                    times.append(t_i)
                    t_i += dt

                return np.array(times)

            else:
                raise ValueError("If t2 < t1, dt must be negative. If t1 < t2, "
                                 "dt should be positive.")

        # dt, t1 : (array_like, numeric)
        elif isinstance(dt, np.ndarray) and t1 is not None:
            times = np.cumsum(np.append([0.], dt)) + t1
            times = times[:-1]

        # nsteps, t1, t2 : (int, numeric, numeric)
        elif dt is None and not (t1 is None or t2 is None or nsteps is None):
            times = np.linspace(t1, t2, nsteps, endpoint=True)

        else:
            raise ValueError("Invalid options. See docstring.")

        return times