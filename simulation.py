
import numpy as np


def simulate_free_vibration(
    system,
    duration,
    dt,
    initial_displacement=0.01,
    initial_velocity=0.0
):
    if duration <= 0 or dt <= 0:
        raise ValueError("Duration and time step must be positive.")

    n_steps = int(np.ceil(duration / dt))

    time = np.linspace(0, duration, n_steps + 1)
    step = time[1] - time[0]

    displacement = np.zeros(n_steps + 1)
    velocity = np.zeros(n_steps + 1)

    state = np.array([
        initial_displacement,
        initial_velocity
    ], dtype=float)

    def derivative(state):
        u, v = state

        acceleration = (
            -system.damping * v
            -system.stiffness * u
        ) / system.mass

        return np.array([v, acceleration])

    displacement[0] = initial_displacement
    velocity[0] = initial_velocity

    for i in range(n_steps):

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step * k1)
        k3 = derivative(state + 0.5 * step * k2)
        k4 = derivative(state + step * k3)

        state = state + (step / 6) * (
            k1 + 2*k2 + 2*k3 + k4
        )

        displacement[i + 1] = state[0]
        velocity[i + 1] = state[1]

    acceleration = (
        -system.damping * velocity
        -system.stiffness * displacement
    ) / system.mass

    return time, displacement, velocity, acceleration

