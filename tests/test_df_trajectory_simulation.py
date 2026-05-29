"""Unit tests for df_trajectory_simulation module.

Tests simulation result containers, error analysis, and performance grading.
"""

import numpy as np
import pytest

from tailsitter.df_trajectory_simulation import (
    SimulationResult,
    ErrorAnalysis,
    analyze_tracking_error,
    _grade,
    _thrust_to_throttle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sim_result(
    n: int = 100,
    dt: float = 0.02,
    V_offset: float = 0.0,
    gamma_offset: float = 0.0,
    theta_offset: float = 0.0,
    q_offset: float = 0.0,
    alpha_offset: float = 0.0,
    h_offset: float = 0.0,
) -> SimulationResult:
    """Create a synthetic SimulationResult with known offsets."""
    t = np.arange(0, n * dt, dt)
    # Reference: simple ramp
    ref_V = np.linspace(0.1, 16.0, len(t))
    ref_gamma = np.linspace(90.0, 0.0, len(t))
    ref_theta = np.linspace(90.0, 5.0, len(t))
    ref_q = np.gradient(ref_theta, t)
    ref_alpha = ref_theta - ref_gamma
    ref_h = np.cumsum(ref_V * np.sin(np.radians(ref_gamma)) * dt)

    return SimulationResult(
        time=t,
        actual_V=ref_V + V_offset,
        actual_gamma=ref_gamma + gamma_offset,
        actual_alpha=ref_alpha + alpha_offset,
        actual_theta=ref_theta + theta_offset,
        actual_q=ref_q + q_offset,
        actual_h=ref_h + h_offset,
        actual_x=np.zeros(len(t)),
        ref_V=ref_V,
        ref_gamma=ref_gamma,
        ref_alpha=ref_alpha,
        ref_theta=ref_theta,
        ref_q=ref_q,
        ref_h=ref_h,
        throttle=np.ones(len(t)) * 0.5,
        elevator=np.ones(len(t)) * 5.0,
    )


# ---------------------------------------------------------------------------
# Tests: dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    def test_simulation_result_defaults(self):
        sr = SimulationResult()
        assert len(sr.time) == 0
        assert len(sr.actual_V) == 0
        assert len(sr.ref_V) == 0

    def test_error_analysis_defaults(self):
        ea = ErrorAnalysis()
        assert ea.rmse_V == 0.0
        assert ea.grade_V == ""


# ---------------------------------------------------------------------------
# Tests: analyze_tracking_error
# ---------------------------------------------------------------------------

class TestAnalyzeTrackingError:
    def test_perfect_tracking(self):
        """When actual == reference, all errors should be zero."""
        sim = _make_sim_result(V_offset=0.0, gamma_offset=0.0,
                               theta_offset=0.0, q_offset=0.0,
                               alpha_offset=0.0, h_offset=0.0)
        error = analyze_tracking_error(sim)

        assert error.rmse_V == pytest.approx(0.0, abs=1e-12)
        assert error.rmse_gamma == pytest.approx(0.0, abs=1e-12)
        assert error.rmse_theta == pytest.approx(0.0, abs=1e-12)
        assert error.rmse_q == pytest.approx(0.0, abs=1e-10)
        assert error.rmse_alpha == pytest.approx(0.0, abs=1e-12)
        assert error.rmse_height == pytest.approx(0.0, abs=1e-12)

        assert error.max_V == pytest.approx(0.0, abs=1e-12)
        assert error.end_V == pytest.approx(0.0, abs=1e-12)

    def test_constant_offset(self):
        """Constant offset should give RMSE = |offset|."""
        offset = 1.5
        sim = _make_sim_result(V_offset=offset)
        error = analyze_tracking_error(sim)

        assert error.rmse_V == pytest.approx(abs(offset), abs=1e-10)
        assert error.max_V == pytest.approx(abs(offset), abs=1e-10)
        assert error.end_V == pytest.approx(offset, abs=1e-10)

    def test_rmse_nonnegative(self):
        """All RMSE values should be non-negative."""
        sim = _make_sim_result(V_offset=0.5, gamma_offset=-0.3)
        error = analyze_tracking_error(sim)

        for attr in ["rmse_V", "rmse_gamma", "rmse_theta", "rmse_q",
                      "rmse_alpha", "rmse_height"]:
            assert getattr(error, attr) >= 0.0

    def test_max_ge_rmse(self):
        """Max deviation should be >= RMSE for each channel."""
        sim = _make_sim_result(V_offset=1.0, theta_offset=0.5)
        error = analyze_tracking_error(sim)

        assert error.max_V >= error.rmse_V - 1e-10
        assert error.max_theta >= error.rmse_theta - 1e-10

    def test_error_fields_populated(self):
        """All ErrorAnalysis fields should be populated."""
        sim = _make_sim_result()
        error = analyze_tracking_error(sim)

        # Check all fields exist and are floats
        for attr in ["rmse_V", "rmse_gamma", "rmse_theta", "rmse_q",
                      "rmse_alpha", "rmse_height",
                      "max_V", "max_gamma", "max_theta", "max_q",
                      "max_alpha", "max_height",
                      "end_V", "end_gamma", "end_theta", "end_q",
                      "end_alpha", "end_height"]:
            assert isinstance(getattr(error, attr), float), f"{attr} not float"

        for attr in ["grade_V", "grade_gamma", "grade_theta", "grade_q"]:
            assert isinstance(getattr(error, attr), str), f"{attr} not str"


# ---------------------------------------------------------------------------
# Tests: performance grading
# ---------------------------------------------------------------------------

class TestGrading:
    def test_grade_V_excellent(self):
        assert _grade(0.3, "V") == "excellent"

    def test_grade_V_good(self):
        assert _grade(0.7, "V") == "good"

    def test_grade_V_fair(self):
        assert _grade(1.5, "V") == "fair"

    def test_grade_V_poor(self):
        assert _grade(3.0, "V") == "poor"

    def test_grade_gamma_thresholds(self):
        assert _grade(0.3, "gamma") == "excellent"
        assert _grade(0.8, "gamma") == "good"
        assert _grade(1.5, "gamma") == "fair"
        assert _grade(3.0, "gamma") == "poor"

    def test_grade_theta_thresholds(self):
        assert _grade(0.5, "theta") == "excellent"
        assert _grade(1.5, "theta") == "good"
        assert _grade(2.5, "theta") == "fair"
        assert _grade(4.0, "theta") == "poor"

    def test_grade_q_thresholds(self):
        assert _grade(1.0, "q") == "excellent"
        assert _grade(3.0, "q") == "good"
        assert _grade(7.0, "q") == "fair"
        assert _grade(15.0, "q") == "poor"

    def test_grade_exact_threshold(self):
        """At exact threshold, should be the higher grade."""
        assert _grade(0.5, "V") == "good"  # not < 0.5
        assert _grade(1.0, "V") == "fair"  # not < 1.0

    def test_error_grades_match_rmse(self):
        """Grades in ErrorAnalysis should match _grade function."""
        sim = _make_sim_result(V_offset=0.3, gamma_offset=0.8,
                               theta_offset=1.5, q_offset=3.0)
        error = analyze_tracking_error(sim)

        assert error.grade_V == _grade(error.rmse_V, "V")
        assert error.grade_gamma == _grade(error.rmse_gamma, "gamma")
        assert error.grade_theta == _grade(error.rmse_theta, "theta")
        assert error.grade_q == _grade(error.rmse_q, "q")


# ---------------------------------------------------------------------------
# Tests: thrust-to-throttle conversion
# ---------------------------------------------------------------------------

class TestThrustToThrottle:
    def test_roundtrip(self):
        """Forward then inverse should recover original throttle."""
        # Simulate aero data
        throttle_arr = np.linspace(0, 1, 50)
        thrust_arr = throttle_arr * 100  # simple linear mapping
        aero_data = {"throttle": throttle_arr, "thrust": thrust_arr}

        # Forward: throttle -> thrust
        original_throttle = 0.6
        thrust_N = 9.8 * np.interp(original_throttle, throttle_arr, thrust_arr)

        # Inverse: thrust -> throttle
        recovered = _thrust_to_throttle(np.array([thrust_N]), aero_data)

        assert recovered[0] == pytest.approx(original_throttle, abs=0.02)

    def test_zero_thrust(self):
        """Zero thrust should give zero throttle."""
        aero_data = {
            "throttle": np.linspace(0, 1, 20),
            "thrust": np.linspace(0, 100, 20),
        }
        result = _thrust_to_throttle(np.array([0.0]), aero_data)
        assert result[0] == pytest.approx(0.0, abs=0.05)

    def test_array_input(self):
        """Should handle array input."""
        aero_data = {
            "throttle": np.linspace(0, 1, 20),
            "thrust": np.linspace(0, 100, 20),
        }
        thrust_arr = np.array([0.0, 9.8 * 50, 9.8 * 100])
        result = _thrust_to_throttle(thrust_arr, aero_data)
        assert len(result) == 3
        assert all(0 <= t <= 1 for t in result)
