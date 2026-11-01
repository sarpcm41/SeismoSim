"""Ground motion inputs: record loading and simple synthetic signals."""

import numpy as np

G = 9.80665 


def load_record(path, dt=None, unit="g"):
    """Load a ground acceleration record from a text or a CSV file.

    Two layouts are supported:
      * two columns  -> (time, acceleration) dt is taken from the time
        column, which must be uniformly and accordingly spaced.
      * one column   -> acceleration only - dt input must be given. :D

    Lines that cannot be read as numbers (headers, comments) are skipped.
    unit is "g" or "m/s2"; the returned acceleration is always in m/s^2.
    """
    if unit not in ("g", "m/s2"):
        raise ValueError('unit must be "g" or "m/s2".')

    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

    rows = [row for row in rows if row]
    if not rows:
        raise ValueError("No numeric data found in the record file.")

    n_columns = len(rows[0])
    if any(len(row) != n_columns for row in rows):
        raise ValueError("Inconsistent number of columns in the record file.")

    data = np.array(rows)
    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Record contains NaN or infinite values."
        )

    if n_columns == 1:
        if dt is None:
            raise ValueError(
                "Single-column record: the time step dt must be provided."
            )
        acceleration = data[:, 0]
    elif n_columns == 2:
        time = data[:, 0]
        steps = np.diff(time)
        if np.any(steps <= 0):
            raise ValueError("Time column must be strictly increasing.")
        if not np.allclose(steps, steps[0], rtol=1e-3):
            raise ValueError("Time column must be uniformly spaced.")
        dt = float(steps[0])
        acceleration = data[:, 1]
    else:
        raise ValueError("Record file must have one or two columns.")

    if len(acceleration) < 2:
        raise ValueError("Record needs at least two samples.")

    if unit == "g":
        acceleration = acceleration * G

    return acceleration, float(dt)


def harmonic(amplitude, frequency_hz, duration, dt):
    """a_g(t) = amplitude * sin(2*pi*f*t), amplitude in m/s^2."""
    _check(duration, dt)
    time = np.arange(int(round(duration / dt)) + 1) * dt
    return amplitude * np.sin(2 * np.pi * frequency_hz * time), dt


def sine_pulse(amplitude, pulse_period, duration, dt):
    """One full sine cycle of the given period, then rest (m/s^2)."""
    _check(duration, dt)
    time = np.arange(int(round(duration / dt)) + 1) * dt
    acceleration = np.where(
        time <= pulse_period,
        amplitude * np.sin(2 * np.pi * time / pulse_period),
        0.0,
    )
    return acceleration, dt


def _check(duration, dt):
    if duration <= 0 or dt <= 0:
        raise ValueError("Duration and time step must be positive.")
