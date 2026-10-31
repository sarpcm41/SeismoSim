"""Elastic response spectra from a ground acceleration record."""

import math

import numpy as np

from model import SDOF
from simulation import simulate_earthquake


def response_spectrum(ground_acceleration, dt, periods, damping_ratio=0.05):
    """Elastic response spectrum for a set of natural periods.

    Parameters
    ----------
    ground_acceleration : record in m/s^2 sampled every dt seconds.
    periods : natural periods (s) to evaluate; all must be positive.
    damping_ratio : critical damping ratio, 0 <= zeta < 1.

    Returns
    -------
    dict with arrays (one value per period):
      "period"                 T (s)
      "Sd"                     peak relative displacement (m)
      "PSv"                    pseudo-velocity, wn * Sd (m/s)
      "PSa"                    pseudo-acceleration, wn^2 * Sd (m/s^2)
      "Sa_abs"                 peak absolute acceleration (m/s^2)
    """
    periods = np.asarray(periods, dtype=float)

    if periods.ndim != 1 or len(periods) == 0:
        raise ValueError("Periods must be a non-empty 1D array.")
    if np.any(periods <= 0):
        raise ValueError("All periods must be positive.")
    if not 0 <= damping_ratio < 1:
        raise ValueError("Damping ratio must satisfy 0 <= zeta < 1.")

    sd = np.zeros(len(periods))
    sa_abs = np.zeros(len(periods))

    for i, period in enumerate(periods):
        wn = 2 * math.pi / period

        # Unit mass: k = wn^2, c = 2*zeta*wn.
        system = SDOF(
            mass=1.0,
            stiffness=wn ** 2,
            damping=2 * damping_ratio * wn,
        )

        _, u, _, _, a_abs = simulate_earthquake(
            system, ground_acceleration, dt
        )

        sd[i] = np.max(np.abs(u))
        sa_abs[i] = np.max(np.abs(a_abs))

    wn_all = 2 * math.pi / periods

    return {
        "period": periods,
        "Sd": sd,
        "PSv": wn_all * sd,
        "PSa": wn_all ** 2 * sd,
        "Sa_abs": sa_abs,
    }
