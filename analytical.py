"""Closed-form solutions used to validate the numerical solvers."""

import numpy as np


def free_vibration_displacement(system, time, u0, v0):
    """Exact free-vibration displacement for any damping level.

    Handles underdamped (zeta < 1), critically damped (zeta == 1)
    and overdamped (zeta > 1) systems.
    """
    time = np.asarray(time, dtype=float)
    wn = system.natural_angular_frequency()
    zeta = system.damping_ratio()

    if np.isclose(zeta, 1.0):
        return np.exp(-wn * time) * (u0 + (v0 + wn * u0) * time)

    if zeta < 1:
        wd = system.damped_angular_frequency()
        alpha = zeta * wn
        b = (v0 + alpha * u0) / wd
        return np.exp(-alpha * time) * (
            u0 * np.cos(wd * time) + b * np.sin(wd * time)
        )

    root = wn * np.sqrt(zeta ** 2 - 1)
    s1 = -zeta * wn + root
    s2 = -zeta * wn - root
    c2 = (v0 - s1 * u0) / (s2 - s1)
    c1 = u0 - c2
    return c1 * np.exp(s1 * time) + c2 * np.exp(s2 * time)


def harmonic_base_steady_state_amplitude(system, ground_amplitude, omega):
    """Steady-state relative displacement amplitude for a ground
    acceleration a_g(t) = ground_amplitude * sin(omega * t)."""
    wn = system.natural_angular_frequency()
    zeta = system.damping_ratio()
    r = omega / wn
    denominator = np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)
    return (ground_amplitude / wn ** 2) / denominator
