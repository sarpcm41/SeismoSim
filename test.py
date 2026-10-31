import os
import tempfile
import unittest

import numpy as np

from analytical import (
    free_vibration_displacement,
    harmonic_base_steady_state_amplitude,
)
from ground_motion import G, harmonic, load_record, sine_pulse
from model import SDOF
from simulation import simulate_earthquake, simulate_free_vibration
from spectrum import response_spectrum


class TestModel(unittest.TestCase):

    def test_str_works_for_overdamped_system(self):
        # zeta = 2 -> previously crashed inside __str__
        system = SDOF(mass=1000, stiffness=200000, damping=57000)
        self.assertGreater(system.damping_ratio(), 1)
        self.assertIn("not defined", str(system))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            SDOF(0, 1, 0)
        with self.assertRaises(ValueError):
            SDOF(1, 0, 0)
        with self.assertRaises(ValueError):
            SDOF(1, 1, -1)


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

        analytical = free_vibration_displacement(
            self.system, time, self.u0, self.v0
        )

        error = np.max(np.abs(u - analytical))
        self.assertLess(error, 1e-8)

    def test_critical_and_overdamped_against_analytical(self):
        for damping in (2 * np.sqrt(200000 * 1000), 60000):
            system = SDOF(1000, 200000, damping)
            time, u, _, _ = simulate_free_vibration(
                system, 2.0, 0.001, 0.01, 0.1
            )
            exact = free_vibration_displacement(system, time, 0.01, 0.1)
            self.assertLess(np.max(np.abs(u - exact)), 1e-8)


class TestEarthquakeSimulation(unittest.TestCase):

    def test_zero_ground_motion_gives_zero_response(self):
        system = SDOF(1000, 200000, 5000)
        _, u, v, a_rel, a_abs = simulate_earthquake(
            system, np.zeros(500), 0.01
        )
        self.assertEqual(np.max(np.abs(u)), 0.0)
        self.assertEqual(np.max(np.abs(a_abs)), 0.0)

    def test_free_vibration_reproduced_with_zero_ground_motion(self):
        system = SDOF(1000, 200000, 10000)
        dt = 0.001
        ground = np.zeros(2001)
        time, u, _, _, _ = simulate_earthquake(
            system, ground, dt, initial_displacement=0.01
        )
        exact = free_vibration_displacement(system, time, 0.01, 0.0)
        self.assertLess(np.max(np.abs(u - exact)), 1e-8)

    def test_harmonic_steady_state_amplitude(self):
        system = SDOF(mass=1.0, stiffness=100.0, damping=2.0)  # zeta = 0.1
        wn = system.natural_angular_frequency()
        omega = 0.8 * wn
        amplitude = 1.0
        dt = 0.002

        ground, _ = harmonic(amplitude, omega / (2 * np.pi), 40.0, dt)
        time, u, _, _, _ = simulate_earthquake(system, ground, dt)

        # Transient decays like exp(-zeta*wn*t); look at the last 5 s.
        steady = u[time > 35.0]
        measured = np.max(np.abs(steady))
        expected = harmonic_base_steady_state_amplitude(
            system, amplitude, omega
        )
        self.assertAlmostEqual(measured / expected, 1.0, delta=1e-3)

    def test_short_period_system_is_stable(self):
        # wn*dt = pi: unstable for plain RK4 without substepping.
        period, dt = 0.02, 0.01
        wn = 2 * np.pi / period
        system = SDOF(1.0, wn ** 2, 2 * 0.05 * wn)
        ground, _ = sine_pulse(1.0, 0.5, 5.0, dt)
        _, u, _, _, _ = simulate_earthquake(system, ground, dt)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertLess(np.max(np.abs(u)), 1.0)

    def test_invalid_inputs(self):
        system = SDOF(1, 1, 0)
        with self.assertRaises(ValueError):
            simulate_earthquake(system, [1.0], 0.01)
        with self.assertRaises(ValueError):
            simulate_earthquake(system, [0.0, 1.0], 0.0)


class TestGroundMotionLoading(unittest.TestCase):

    def _write(self, text):
        handle, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_two_column_record_in_g(self):
        path = self._write("time,acc\n0.0,0.0\n0.01,0.1\n0.02,-0.1\n")
        acc, dt = load_record(path, unit="g")
        self.assertAlmostEqual(dt, 0.01)
        np.testing.assert_allclose(acc, [0.0, 0.1 * G, -0.1 * G])

    def test_single_column_needs_dt(self):
        path = self._write("0.0\n1.0\n2.0\n")
        with self.assertRaises(ValueError):
            load_record(path)
        acc, dt = load_record(path, dt=0.02, unit="m/s2")
        self.assertEqual(dt, 0.02)
        np.testing.assert_allclose(acc, [0.0, 1.0, 2.0])

    def test_non_uniform_time_rejected(self):
        path = self._write("0.0 0.0\n0.01 1.0\n0.05 2.0\n")
        with self.assertRaises(ValueError):
            load_record(path)


class TestResponseSpectrum(unittest.TestCase):

    def test_rigid_and_flexible_limits(self):
        dt = 0.005
        ground, _ = sine_pulse(2.0, 0.4, 8.0, dt)
        pga = np.max(np.abs(ground))

        # Very stiff structure follows the ground: Sa_abs -> PGA.
        # Very flexible structure stays still: Sd -> peak ground displacement.
        spec = response_spectrum(ground, dt, [0.02, 30.0], damping_ratio=0.05)
        self.assertAlmostEqual(spec["Sa_abs"][0] / pga, 1.0, delta=0.05)

        velocity = np.cumsum(ground) * dt
        displacement = np.cumsum(velocity) * dt
        pgd = np.max(np.abs(displacement))
        self.assertGreater(spec["Sd"][1], 0.5 * pgd)

    def test_pseudo_acceleration_relation(self):
        dt = 0.01
        ground, _ = sine_pulse(1.0, 0.5, 4.0, dt)
        periods = np.array([0.2, 0.5, 1.0])
        spec = response_spectrum(ground, dt, periods)
        wn = 2 * np.pi / periods
        np.testing.assert_allclose(spec["PSa"], wn ** 2 * spec["Sd"])
        np.testing.assert_allclose(spec["PSv"], wn * spec["Sd"])

    def test_invalid_inputs(self):
        ground = np.zeros(10)
        with self.assertRaises(ValueError):
            response_spectrum(ground, 0.01, [0.0])
        with self.assertRaises(ValueError):
            response_spectrum(ground, 0.01, [1.0], damping_ratio=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
