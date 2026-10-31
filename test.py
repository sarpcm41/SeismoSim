import unittest
import numpy as np

from model import SDOF
from simulation import simulate_free_vibration


class TestFreeVibrationSimulation(unittest.TestCase):

    def setUp(self):
        self.system = SDOF(
            mass=1000,
            stiffness=200000,
            damping=10000
        )

        self.u0 = 0.01
        self.v0 = 0.0
        self.duration = 2.0
        self.dt = 0.001

    def test_initial_conditions(self):
        time, u, v, a = simulate_free_vibration(
            self.system,
            self.duration,
            self.dt,
            self.u0,
            self.v0
        )

        self.assertAlmostEqual(u[0], self.u0)
        self.assertAlmostEqual(v[0], self.v0)

    def test_initial_acceleration(self):
        time, u, v, a = simulate_free_vibration(
            self.system,
            self.duration,
            self.dt,
            self.u0,
            self.v0
        )

        expected_acceleration = (
            -self.system.damping * self.v0
            -self.system.stiffness * self.u0
        ) / self.system.mass

        self.assertAlmostEqual(
            a[0],
            expected_acceleration
        )

    def test_rk4_against_analytical_solution(self):
        time, u, v, a = simulate_free_vibration(
            self.system,
            self.duration,
            self.dt,
            self.u0,
            self.v0
        )

        wn = self.system.natural_angular_frequency()
        zeta = self.system.damping_ratio()
        wd = self.system.damped_angular_frequency()

        alpha = zeta * wn

        B = (
            self.v0 + alpha * self.u0
        ) / wd

        analytical_displacement = (
            np.exp(-alpha * time)
            * (
                self.u0 * np.cos(wd * time)
                + B * np.sin(wd * time)
            )
        )

        error = np.max(
            np.abs(u - analytical_displacement)
        )

        self.assertLess(error, 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
