
from model import SDOF
from simulation import simulate_free_vibration
from visualization import plot_free_vibration

import numpy as np

from simulation import simulate_earthquake
from ground_motion import load_record
from spectrum import response_spectrum
from visualization import plot_earthquake_response, plot_response_spectrum


def main():

    print("=== SeismoSim ===")
    print("Free Vibration Simulation\n")

    try:
        mass = float(input("Mass (kg): "))
        stiffness = float(input("Stiffness (N/m): "))
        damping = float(input("Damping (Ns/m): "))

        system = SDOF(mass, stiffness, damping)

        initial_displacement = float(
            input("Initial displacement (m) [0.01]: ") or 0.01
        )

        initial_velocity = float(
            input("Initial velocity (m/s) [0.0]: ") or 0.0
        )

        duration = float(input("Simulation duration (s) [2.0]: ") or 2.0)

        dt = float(input("Time step (s) [0.001]: ") or 0.001)

        time, displacement, velocity, acceleration = (
            simulate_free_vibration(
                system,
                duration,
                dt,
                initial_displacement,
                initial_velocity
            )
        )

    except ValueError as error:
        print(f"\nError: {error}")
        return

    print("\n=== System Properties ===")
    print(system)

    print("=== Simulation Results ===")
    print(f"Maximum displacement: {max(abs(displacement)):.6f} m")
    print(f"Maximum velocity: {max(abs(velocity)):.6f} m/s")
    print(f"Maximum acceleration: {max(abs(acceleration)):.6f} m/s²")

    plot_free_vibration(
        time,
        displacement,
        velocity,
        acceleration
    )


def ask_float(prompt, default=None):
    text = input(prompt).strip()
    if text == "" and default is not None:
        return default
    return float(text)


def ask_system():
    mass = ask_float("Mass (kg): ")
    stiffness = ask_float("Stiffness (N/m): ")
    damping = ask_float("Damping (Ns/m): ")
    return SDOF(mass, stiffness, damping)


def ask_record():
    path = input("Record file path (CSV/TXT): ").strip()
    unit = input("Acceleration unit, g or m/s2 [g]: ").strip() or "g"
    dt_text = input("Time step (s) [read from file]: ").strip()
    dt = float(dt_text) if dt_text else None
    return load_record(path, dt=dt, unit=unit)


def run_earthquake():

    print("\nEarthquake Response\n")

    system = ask_system()
    ground_acceleration, dt = ask_record()

    time, u, v, _, a_abs = simulate_earthquake(
        system,
        ground_acceleration,
        dt
    )

    print("\n=== System Properties ===")
    print(system)

    print("=== Earthquake Results ===")
    print(f"Peak ground acceleration: "
          f"{np.max(np.abs(ground_acceleration)):.4f} m/s²")
    print(f"Maximum relative displacement: {np.max(np.abs(u)):.6f} m")
    print(f"Maximum relative velocity: {np.max(np.abs(v)):.6f} m/s")
    print(f"Maximum absolute acceleration: "
          f"{np.max(np.abs(a_abs)):.4f} m/s²")

    plot_earthquake_response(time, ground_acceleration, u, v, a_abs)


def run_spectrum():

    print("\nResponse Spectrum\n")

    ground_acceleration, dt = ask_record()
    zeta = ask_float("Damping ratio [0.05]: ", 0.05)
    t_min = ask_float("Minimum period (s) [0.05]: ", 0.05)
    t_max = ask_float("Maximum period (s) [4.0]: ", 4.0)
    count = int(ask_float("Number of periods [100]: ", 100))

    periods = np.linspace(t_min, t_max, count)
    spectrum = response_spectrum(ground_acceleration, dt, periods, zeta)

    peak = np.argmax(spectrum["PSa"])
    print("\n=== Spectrum Results ===")
    print(f"Peak pseudo-acceleration: {spectrum['PSa'][peak]:.4f} m/s² "
          f"at T = {spectrum['period'][peak]:.3f} s")

    plot_response_spectrum(spectrum, zeta)


def menu():

    print("=== SeismoSim ===")
    print("1) Free vibration")
    print("2) Earthquake response (ground motion record)")
    print("3) Response spectrum (ground motion record)")

    choice = input("\nSelect [1]: ").strip() or "1"

    actions = {
        "1": main,
        "2": run_earthquake,
        "3": run_spectrum,
    }

    action = actions.get(choice)

    if action is None:
        print("Invalid selection.")
        return

    try:
        action()
    except (ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    menu()
