"""Integration tests for SeismoSim. pls work

Groups:
  A. Smoke           - every module imports
  B. End to end      - file -> simulation -> spectrum chains, unit chain
  C. Physics         - linearity, superposition, energy, log decrement ...
  D. Cross-check     - independent Newmark beta solver, convergence order
  E. Edge cases      - zeta = 0 / 1 / >1, very short periods, bad files
  F. CLI / menu      - input() and plt.show() are mocked

Run:  py test_integration.py      (or: py -m unittest -v)
"""

import contextlib
import importlib
import io
import os
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analytical import free_vibration_displacement
from ground_motion import G, harmonic, load_record, sine_pulse
from model import SDOF
from simulation import simulate_earthquake, simulate_free_vibration
from spectrum import response_spectrum


def write_temp(testcase, text, suffix=".csv"):
    handle, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(text)
    testcase.addCleanup(os.remove, path)
    return path


def two_column_text(acc, dt, scale=1.0):
    lines = ["time,acc"]
    for i, a in enumerate(acc):
        lines.append(f"{i * dt:.6f},{a * scale:.12e}")
    return "\n".join(lines) + "\n"


def newmark_average_acceleration(m, c, k, p, dt, u0=0.0, v0=0.0):
    """Independent reference solver (Newmark-beta, average acceleration)."""
    beta, gamma = 0.25, 0.5
    n = len(p)
    u = np.zeros(n)
    v = np.zeros(n)
    u[0], v[0] = u0, v0
    a = (p[0] - c * v0 - k * u0) / m

    k_hat = k + gamma / (beta * dt) * c + m / (beta * dt ** 2)
    coef_a = m / (beta * dt) + gamma / beta * c
    coef_b = m / (2 * beta) + dt * (gamma / (2 * beta) - 1) * c

    for i in range(n - 1):
        dp = p[i + 1] - p[i] + coef_a * v[i] + coef_b * a
        du = dp / k_hat
        dv = (gamma / (beta * dt) * du
              - gamma / beta * v[i]
              + dt * (1 - gamma / (2 * beta)) * a)
        u[i + 1] = u[i] + du
        v[i + 1] = v[i] + dv
        a = (p[i + 1] - c * v[i + 1] - k * u[i + 1]) / m
    return u, v


# ---------------------------------------------------------------- A. smoke
class TestSmoke(unittest.TestCase):

    def test_all_modules_import(self):
        for name in (
            "model", "simulation", "ground_motion", "spectrum",
            "analytical", "visualization", "converge", "main",
        ):
            with self.subTest(module=name):
                importlib.import_module(name)


# ------------------------------------------------------------ B. end to end
class TestEndToEnd(unittest.TestCase):

    def test_file_to_spectrum_chain(self):
        dt = 0.01
        original, _ = sine_pulse(2.0, 0.5, 6.0, dt)
        path = write_temp(self, two_column_text(original, dt, 1.0 / G))

        acc, dt_read = load_record(path, unit="g")
        self.assertAlmostEqual(dt_read, dt)
        np.testing.assert_allclose(acc, original, rtol=1e-9, atol=1e-9)

        system = SDOF(1000, 200000, 5000)
        time, u, v, a_rel, a_abs = simulate_earthquake(system, acc, dt_read)
        for arr in (time, u, v, a_rel, a_abs):
            self.assertEqual(len(arr), len(acc))
            self.assertTrue(np.all(np.isfinite(arr)))

        periods = np.array([0.1, 0.5, 1.0, 2.0])
        spec = response_spectrum(acc, dt_read, periods, 0.05)
        for key in ("period", "Sd", "PSv", "PSa", "Sa_abs"):
            self.assertEqual(len(spec[key]), len(periods))
            self.assertTrue(np.all(np.isfinite(spec[key])))
        self.assertTrue(np.all(spec["Sd"] > 0))

    def test_spectrum_matches_direct_simulation(self):
        dt, zeta, period = 0.01, 0.05, 0.5
        ground, _ = sine_pulse(1.5, 0.6, 5.0, dt)

        spec = response_spectrum(ground, dt, [period], zeta)

        wn = 2 * np.pi / period
        system = SDOF(1.0, wn ** 2, 2 * zeta * wn)
        _, u, _, _, a_abs = simulate_earthquake(system, ground, dt)

        self.assertAlmostEqual(spec["Sd"][0], np.max(np.abs(u)), places=12)
        self.assertAlmostEqual(spec["Sa_abs"][0], np.max(np.abs(a_abs)),
                               places=10)

    def test_unit_chain_g_vs_ms2(self):
        dt = 0.01
        ground, _ = sine_pulse(1.0, 0.5, 4.0, dt)
        path = write_temp(
            self, "\n".join(f"{a:.12e}" for a in ground) + "\n"
        )
        acc_g, _ = load_record(path, dt=dt, unit="g")
        acc_ms2, _ = load_record(path, dt=dt, unit="m/s2")
        np.testing.assert_allclose(acc_g, acc_ms2 * G, rtol=1e-12)

        system = SDOF(1000, 200000, 5000)
        _, u_g, _, _, _ = simulate_earthquake(system, acc_g, dt)
        _, u_ms2, _, _, _ = simulate_earthquake(system, acc_ms2, dt)
        ratio = np.max(np.abs(u_g)) / np.max(np.abs(u_ms2))
        self.assertAlmostEqual(ratio, G, places=6)


# --------------------------------------------------------------- C. physics
class TestPhysics(unittest.TestCase):

    def setUp(self):
        self.system = SDOF(1000, 200000, 5000)
        self.dt = 0.01
        self.g1, _ = sine_pulse(2.0, 0.5, 5.0, self.dt)
        self.g2, _ = harmonic(1.0, 3.0, 5.0, self.dt)

    def test_linearity(self):
        _, u1, v1, _, a1 = simulate_earthquake(self.system, self.g1, self.dt)
        _, u2, v2, _, a2 = simulate_earthquake(
            self.system, 2.0 * self.g1, self.dt
        )
        np.testing.assert_allclose(u2, 2 * u1, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(v2, 2 * v1, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a2, 2 * a1, rtol=1e-9, atol=1e-9)

    def test_superposition(self):
        _, u1, _, _, _ = simulate_earthquake(self.system, self.g1, self.dt)
        _, u2, _, _, _ = simulate_earthquake(self.system, self.g2, self.dt)
        _, u12, _, _, _ = simulate_earthquake(
            self.system, self.g1 + self.g2, self.dt
        )
        np.testing.assert_allclose(u12, u1 + u2, rtol=1e-9, atol=1e-12)

    def test_mass_independence(self):
        # Same wn and zeta, different mass -> same relative displacement.
        light = SDOF(1.0, 100.0, 2.0)
        heavy = SDOF(50.0, 5000.0, 100.0)
        self.assertAlmostEqual(light.damping_ratio(), heavy.damping_ratio())
        _, u_light, _, _, _ = simulate_earthquake(light, self.g1, self.dt)
        _, u_heavy, _, _, _ = simulate_earthquake(heavy, self.g1, self.dt)
        np.testing.assert_allclose(u_light, u_heavy, rtol=1e-9, atol=1e-12)

    def test_undamped_energy_conservation(self):
        system = SDOF(1000, 200000, 0.0)
        _, u, v, _ = simulate_free_vibration(system, 2.0, 0.001, 0.01, 0.1)
        energy = (0.5 * system.mass * v ** 2
                  + 0.5 * system.stiffness * u ** 2)
        drift = np.max(np.abs(energy - energy[0])) / energy[0]
        self.assertLess(drift, 1e-6)

    def test_undamped_pseudo_acceleration_equals_absolute(self):
        periods = np.array([0.2, 0.5, 1.0])
        spec = response_spectrum(self.g1, self.dt, periods, damping_ratio=0.0)
        np.testing.assert_allclose(spec["PSa"], spec["Sa_abs"], rtol=1e-9)

    def test_log_decrement_recovers_damping_ratio(self):
        system = SDOF(1.0, 100.0, 1.0)  # zeta = 0.05
        zeta = system.damping_ratio()
        _, u, _, _ = simulate_free_vibration(system, 10.0, 0.001, 1.0, 0.0)

        peaks = [
            u[i] for i in range(1, len(u) - 1)
            if u[i] > 0 and u[i] > u[i - 1] and u[i] >= u[i + 1]
        ]
        self.assertGreater(len(peaks), 6)
        deltas = [np.log(peaks[i] / peaks[i + 1]) for i in range(5)]
        delta = np.mean(deltas)
        zeta_est = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)
        self.assertAlmostEqual(zeta_est, zeta, delta=1e-3)

    def test_resonance_peak_at_drive_period(self):
        drive_hz = 2.0  # T = 0.5 s
        dt = 0.005
        ground, _ = harmonic(1.0, drive_hz, 20.0, dt)
        periods = np.linspace(0.4, 0.6, 41)
        spec = response_spectrum(ground, dt, periods, 0.05)
        peak_period = spec["period"][np.argmax(spec["PSa"])]
        self.assertAlmostEqual(peak_period, 1.0 / drive_hz, delta=0.02)

    def test_time_step_independence(self):
        system = SDOF(1.0, 4 * np.pi ** 2 / 0.25, 2 * 0.05 * 2 * np.pi / 0.5)
        coarse_dt, fine_dt = 0.01, 0.005
        g_coarse, _ = harmonic(1.0, 1.0, 10.0, coarse_dt)
        g_fine, _ = harmonic(1.0, 1.0, 10.0, fine_dt)

        _, u_coarse, _, _, _ = simulate_earthquake(system, g_coarse, coarse_dt)
        _, u_fine, _, _, _ = simulate_earthquake(system, g_fine, fine_dt)

        diff = np.max(np.abs(u_coarse - u_fine[::2]))
        self.assertLess(diff / np.max(np.abs(u_fine)), 1e-2)


# ---------------------------------------------------------- D. cross-checks
class TestCrossValidation(unittest.TestCase):

    def test_rk4_matches_independent_newmark_solver(self):
        system = SDOF(2.0, 2.0 * 160.0, 2.0 * 3.0)
        dt = 0.002
        ground, _ = sine_pulse(2.0, 0.5, 5.0, dt)

        _, u_rk4, v_rk4, _, _ = simulate_earthquake(system, ground, dt)
        u_nm, v_nm = newmark_average_acceleration(
            system.mass, system.damping, system.stiffness,
            -system.mass * ground, dt,
        )
        scale = np.max(np.abs(u_nm))
        self.assertLess(np.max(np.abs(u_rk4 - u_nm)) / scale, 1e-2)

    def test_rk4_convergence_order_is_about_four(self):
        system = SDOF(1000, 200000, 10000)
        u0, v0 = 0.01, 0.0
        time_steps = [0.02, 0.01, 0.005, 0.0025]
        errors = []
        for dt in time_steps:
            time, u, _, _ = simulate_free_vibration(system, 2.0, dt, u0, v0)
            exact = free_vibration_displacement(system, time, u0, v0)
            errors.append(np.max(np.abs(u - exact)))

        order = np.polyfit(np.log(time_steps), np.log(errors), 1)[0]
        self.assertGreater(order, 3.6)
        self.assertLess(order, 4.4)
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])))


# ------------------------------------------------------------ E. edge cases
class TestEdgeCases(unittest.TestCase):

    def test_all_damping_regimes_stay_finite(self):
        dt = 0.01
        ground, _ = sine_pulse(1.0, 0.5, 5.0, dt)
        critical = 2 * np.sqrt(200000 * 1000)
        for damping in (0.0, critical, 60000.0):
            with self.subTest(damping=damping):
                system = SDOF(1000, 200000, damping)
                _, u, v, a_rel, a_abs = simulate_earthquake(
                    system, ground, dt
                )
                for arr in (u, v, a_rel, a_abs):
                    self.assertTrue(np.all(np.isfinite(arr)))

    def test_very_short_and_very_long_periods(self):
        dt = 0.01
        ground, _ = sine_pulse(1.0, 0.5, 5.0, dt)
        spec = response_spectrum(ground, dt, [0.01, 0.05, 20.0], 0.05)
        for key in ("Sd", "PSv", "PSa", "Sa_abs"):
            self.assertTrue(np.all(np.isfinite(spec[key])))

    def test_bad_record_files_are_rejected(self):
        cases = {
            "empty": "",
            "header only": "time,acc\n",
            "three columns": "0 0 0\n0.01 1 1\n0.02 2 2\n",
            "inconsistent columns": "0.0 1.0\n0.01\n0.02 3.0\n",
            "decreasing time": "0.0 0.0\n0.02 1.0\n0.01 2.0\n",
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                path = write_temp(self, text)
                with self.assertRaises(ValueError):
                    load_record(path)

    def test_missing_record_file(self):
        with self.assertRaises(OSError):
            load_record("this_file_does_not_exist.csv")

    def test_long_record_runs_in_reasonable_time(self):
        import time as clock
        dt = 0.01
        ground, _ = harmonic(1.0, 1.0, 100.0, dt)  # 10,001 samples
        start = clock.perf_counter()
        spec = response_spectrum(ground, dt, np.linspace(0.1, 2.0, 20), 0.05)
        elapsed = clock.perf_counter() - start
        self.assertTrue(np.all(np.isfinite(spec["Sd"])))
        self.assertLess(elapsed, 60.0)


# ------------------------------------------------------------- F. CLI / menu
class TestMenu(unittest.TestCase):

    def run_menu(self, inputs):
        import main as app
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=list(inputs)), \
                mock.patch("matplotlib.pyplot.show") as show, \
                contextlib.redirect_stdout(out):
            app.menu()
        return out.getvalue(), show

    def record_path(self):
        dt = 0.01
        ground, _ = sine_pulse(2.0, 0.5, 6.0, dt)
        return write_temp(self, two_column_text(ground, dt, 1.0 / G))

    def test_option_1_free_vibration(self):
        text, show = self.run_menu(
            ["1", "1000", "200000", "10000", "", "", "", ""]
        )
        self.assertIn("Maximum displacement", text)
        show.assert_called_once()

    def test_empty_choice_defaults_to_free_vibration(self):
        text, show = self.run_menu(
            ["", "1000", "200000", "10000", "", "", "", ""]
        )
        self.assertIn("Simulation Results", text)
        show.assert_called_once()

    def test_option_2_earthquake_response(self):
        path = self.record_path()
        text, show = self.run_menu(
            ["2", "1000", "200000", "5000", path, "g", ""]
        )
        self.assertIn("Maximum relative displacement", text)
        show.assert_called_once()

    def test_option_3_response_spectrum(self):
        path = self.record_path()
        text, show = self.run_menu(
            ["3", path, "g", "", "", "0.1", "1.0", "10"]
        )
        self.assertIn("Peak pseudo-acceleration", text)
        show.assert_called_once()

    def test_invalid_selection(self):
        text, show = self.run_menu(["9"])
        self.assertIn("Invalid selection", text)
        show.assert_not_called()

    def test_bad_inputs_do_not_crash(self):
        cases = {
            "letters instead of number": ["1", "abc"],
            "negative mass": ["1", "-5", "200000", "10000"],
            "missing file": ["2", "1", "1", "0", "no_such_file.csv", "g", ""],
        }
        for name, inputs in cases.items():
            with self.subTest(case=name):
                text, show = self.run_menu(inputs)
                self.assertIn("Error", text)
                show.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
