
from model import SDOF


def main():
    print("=== SeismoSim ===")
    print("Single Degree of Freedom System\n")

    mass = float(input("Enter mass (kg): "))
    stiffness = float(input("Enter stiffness (N/m): "))
    damping = float(input("Enter damping (Ns/m): "))

    system = SDOF(mass, stiffness, damping)

    print("\n=== System Properties ===")

    print(f"Natural Angular Frequency: {system.natural_angular_frequency():.4f} rad/s")
    print(f"Natural Period: {system.natural_period():.4f} s")
    print(f"Critical Damping: {system.critical_damping():.4f} Ns/m")
    print(f"Damping Ratio: {system.damping_ratio():.4f}")

    zeta = system.damping_ratio()

    if zeta < 1:
        print(f"Damped Angular Frequency: {system.damped_angular_frequency():.4f} rad/s")
        print(f"Damped Period: {system.damped_period():.4f} s")
    else:
        print("Damped frequency: Not applicable for zeta >= 1")


if __name__ == "__main__":
    main()

