
from model import SDOF
from simulation import simulate_free_vibration
from visualization import plot_free_vibration


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


if __name__ == "__main__":
    main()
