
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


