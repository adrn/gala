"""Helper function for turning different ways of specifying the integration
times into an array of times.
"""

import numpy as np

__all__ = ["parse_time_specification"]


def parse_time_specification(units, dt=None, n_steps=None, t1=None, t2=None, t=None):
    """
    Return an array of times given a few combinations of kwargs that are
    accepted -- see below.

    Parameters
    ----------
    dt, n_steps[, t1] : (numeric, int[, numeric])
        A fixed timestep dt and a number of steps to run for.
    dt, t1, t2 : (numeric, numeric, numeric)
        A fixed timestep dt, an initial time, and an final time.
    dt, t1 : (array_like, numeric)
        An array of timesteps dt and an initial time.
    n_steps, t1, t2 : (int, numeric, numeric)
        Number of steps between an initial time, and a final time.
    t : array_like
        An array of times (dts = t[1:] - t[:-1])

    """
    if n_steps is not None:  # parse and validate n_steps
        n_steps = int(n_steps)

    if hasattr(dt, "unit"):
        dt = dt.decompose(units).value

    if hasattr(t1, "unit"):
        t1 = t1.decompose(units).value

    if hasattr(t2, "unit"):
        t2 = t2.decompose(units).value

    if hasattr(t, "unit"):
        t = t.decompose(units).value

    # t : array_like
    if t is not None:
        times = t
        return times.astype(np.float64)

    if dt is None and (t1 is None or t2 is None or n_steps is None):
        raise ValueError(
            "Invalid specification of integration time. See docstring for more "
            "information."
        )

    # dt, n_steps[, t1] : (numeric, int[, numeric])
    if dt is not None and n_steps is not None:
        if t1 is None:
            t1 = 0.0

        times = parse_time_specification(units, dt=np.ones(n_steps + 1) * dt, t1=t1)

    # dt, t1, t2 : (numeric, numeric, numeric)
    elif dt is not None and t1 is not None and t2 is not None:
        if t2 < t1 and dt < 0:
            t_i = t1
            times = []
            ii = 0
            while (t_i > t2) and (ii < 1e6):
                times.append(t_i)
                t_i += dt

            if times[-1] != t2:
                times.append(t2)

            return np.array(times, dtype=np.float64)

        if t2 > t1 and dt > 0:
            t_i = t1
            times = []
            ii = 0
            while (t_i < t2) and (ii < 1e6):
                times.append(t_i)
                t_i += dt

            return np.array(times, dtype=np.float64)

        if dt == 0:
            raise ValueError("dt must be non-zero.")
        raise ValueError(
            "If t2 < t1, dt must be negative. If t1 < t2, dt must be positive."
        )

    # dt, t1 : (array_like, numeric)
    elif isinstance(dt, np.ndarray) and t1 is not None:
        times = np.cumsum(np.append([0.0], dt)) + t1
        times = times[:-1]

    # n_steps, t1, t2 : (int, numeric, numeric)
    elif dt is None and not (t1 is None or t2 is None or n_steps is None):
        times = np.linspace(t1, t2, n_steps, endpoint=True)

    else:
        raise ValueError("Invalid options. See docstring.")

    return times.astype(np.float64)
