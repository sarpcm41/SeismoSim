import matplotlib.pyplot as plt


def plot_free_vibration(time, displacement, velocity, acceleration):

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 8),
        sharex=True
    )

    axes[0].plot(time, displacement)
    axes[0].set_ylabel("Displacement (m)")
    axes[0].set_title("Free Vibration Response")
    axes[0].grid(True)

    axes[1].plot(time, velocity)
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True)

    axes[2].plot(time, acceleration)
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_earthquake_response(
    time,
    ground_acceleration,
    displacement,
    velocity,
    absolute_acceleration
):

    fig, axes = plt.subplots(
        4, 1,
        figsize=(10, 10),
        sharex=True
    )

    axes[0].plot(time, ground_acceleration, color="gray")
    axes[0].set_ylabel("Ground acc (m/s²)")
    axes[0].set_title("Earthquake Response")
    axes[0].grid(True)

    axes[1].plot(time, displacement)
    axes[1].set_ylabel("Rel. displacement (m)")
    axes[1].grid(True)

    axes[2].plot(time, velocity)
    axes[2].set_ylabel("Rel. velocity (m/s)")
    axes[2].grid(True)

    axes[3].plot(time, absolute_acceleration)
    axes[3].set_ylabel("Abs. acceleration (m/s²)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def plot_response_spectrum(spectrum, damping_ratio):

    period = spectrum["period"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    axes[0].plot(period, spectrum["PSa"])
    axes[0].set_xlabel("Period (s)")
    axes[0].set_ylabel("Pseudo acceleration (m/s²)")
    axes[0].set_title(
        f"Acceleration Spectrum (ζ = {damping_ratio:.2f})"
    )
    axes[0].grid(True)

    axes[1].plot(period, spectrum["Sd"])
    axes[1].set_xlabel("Period (s)")
    axes[1].set_ylabel("Displacement (m)")
    axes[1].set_title(
        f"Displacement Spectrum (ζ = {damping_ratio:.2f})"
    )
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
