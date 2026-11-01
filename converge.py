"""RK4 convergence study against the closed-form free vibration solution."""

import matplotlib.pyplot as plt
import numpy as np

from analytical import free_vibration_displacement
from model import SDOF
from simulation import simulate_free_vibration


def main():

    # Physical system
    system = SDOF(
        mass=1000,
        stiffness=200000,
        damping=10000
    )

    # Initial conditions
    u0 = 0.01
    v0 = 0.0

    duration = 2.0

    # Different time steps
    time_steps = [
        0.02,
        0.01,
        0.005,
        0.0025,
        0.00125
    ]

    errors = []

    print("=== RK4 Convergence Analysis ===\n")

    print(
        f"{'Time Step':>12}"
        f"{'Maximum Error':>20}"
        f"{'Error Ratio':>16}"
        f"{'Observed Order':>18}"
    )

    print("-" * 68)

    for dt in time_steps:

        # Numerical simulation
        time, u, v, a = simulate_free_vibration(
            system,
            duration,
            dt,
            u0,
            v0
        )

        # Analytical solution
        exact = free_vibration_displacement(
            system,
            time,
            u0,
            v0
        )

        # Maximum absolute error
        error = np.max(np.abs(u - exact))

        errors.append(error)

    # Calculate error ratios and observed orders
    for i, dt in enumerate(time_steps):

        if i == 0:
            ratio_text = "-"
            order_text = "-"
        else:
            ratio = errors[i - 1] / errors[i]
            order = np.log2(ratio)

            ratio_text = f"{ratio:.4f}"
            order_text = f"{order:.4f}"

        print(
            f"{dt:12.5f}"
            f"{errors[i]:20.10e}"
            f"{ratio_text:>16}"
            f"{order_text:>18}"
        )

    # Estimate overall convergence order
    observed_order = np.polyfit(
        np.log(time_steps),
        np.log(errors),
        1
    )[0]

    print("\n=== Analysis ===")
    print(f"Overall observed order: {observed_order:.4f}")
    print("Theoretical RK4 order: 4")

    # Plot convergence
    dt_array = np.array(time_steps)
    error_array = np.array(errors)

    reference_error = (
        errors[0]
        * (dt_array / dt_array[0]) ** 4
    )

    plt.figure(figsize=(9, 6))

    plt.loglog(
        dt_array,
        error_array,
        "o-",
        label="RK4 numerical error"
    )

    plt.loglog(
        dt_array,
        reference_error,
        "--",
        label="Fourth-order reference"
    )

    plt.xlabel("Time Step (s)")
    plt.ylabel("Maximum Absolute Error (m)")

    plt.title("RK4 Convergence Analysis")

    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
