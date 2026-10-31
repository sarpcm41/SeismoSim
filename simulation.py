

# Largest allowed value of (natural frequency * internal step) for RK4.
# RK4 is only stable for roughly wn*h < 2.8; staying well below keeps it
# accurate too, which matters for stiff (short-period) systems.
MAX_FREQUENCY_STEP_PRODUCT = 0.1


def _rk4_base_excitation(c_over_m, k_over_m, ground_acc, dt, u0, v0, substeps):
    """RK4 integration of  u'' + (c/m) u' + (k/m) u = -a_g(t).

    ground_acc is sampled every dt. Inside each sampling interval the
    ground acceleration is linearly interpolated, which makes the value
    used at the RK4 midpoint exact for that interpolation.
    """
    n = len(ground_acc) - 1
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    u[0], v[0] = u0, v0

    h = dt / substeps
    cu, ku = c_over_m, k_over_m
    ui, vi = float(u0), float(v0)

    for i in range(n):
        a_start = ground_acc[i]
        a_end = ground_acc[i + 1]

        for j in range(substeps):
            a0 = a_start + (a_end - a_start) * (j / substeps)
            a1 = a_start + (a_end - a_start) * ((j + 1) / substeps)
            am = 0.5 * (a0 + a1)

            k1u = vi
            k1v = -cu * vi - ku * ui - a0

            k2u = vi + 0.5 * h * k1v
            k2v = -cu * k2u - ku * (ui + 0.5 * h * k1u) - am

            k3u = vi + 0.5 * h * k2v
            k3v = -cu * k3u - ku * (ui + 0.5 * h * k2u) - am

            k4u = vi + h * k3v
            k4v = -cu * k4u - ku * (ui + h * k3u) - a1

            ui += (h / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
            vi += (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

        u[i + 1] = ui
        v[i + 1] = vi

    return u, v


def simulate_earthquake(
    system,
    ground_acceleration,
    dt,
    initial_displacement=0.0,
    initial_velocity=0.0
):
    """Response of an SDOF system to a ground acceleration record.

    Parameters
    ----------
    ground_acceleration : array of ground accelerations in m/s^2,
        sampled at a constant interval dt (s).

    Returns
    -------
    time, displacement, velocity, relative_acceleration,
    absolute_acceleration

    displacement and velocity are relative to the ground. The absolute
    acceleration is the relative acceleration plus the ground
    acceleration, i.e. what an accelerometer on the structure measures.
    """
    ground_acceleration = np.asarray(ground_acceleration, dtype=float)

    if ground_acceleration.ndim != 1 or len(ground_acceleration) < 2:
        raise ValueError(
            "Ground acceleration must be a 1D array with at least 2 samples."
        )
    if dt <= 0:
        raise ValueError("Time step must be positive.")

    wn = system.natural_angular_frequency()
    substeps = max(1, int(np.ceil(wn * dt / MAX_FREQUENCY_STEP_PRODUCT)))

    c_over_m = system.damping / system.mass
    k_over_m = system.stiffness / system.mass

    u, v = _rk4_base_excitation(
        c_over_m,
        k_over_m,
        ground_acceleration,
        dt,
        initial_displacement,
        initial_velocity,
        substeps
    )

    time = np.arange(len(ground_acceleration)) * dt

    absolute_acceleration = -(c_over_m * v + k_over_m * u)
    relative_acceleration = absolute_acceleration - ground_acceleration

    return time, u, v, relative_acceleration, absolute_acceleration
