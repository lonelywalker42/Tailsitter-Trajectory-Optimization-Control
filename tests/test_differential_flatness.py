"""Tests for differential flatness transfer module."""

import numpy as np
import pytest
from pathlib import Path

from tailsitter.differential_flatness import (
    AeroSin4Coeffs,
    DFTransferResult,
    PitchRateMargin,
    PitchRateMarginParams,
    aero_sin4_eval,
    build_sin4_coeffs,
    differential_flatness_transfer,
    fit_sin4,
    load_sin4_coeffs,
    calculate_max_feasible_pitch_rate,
)
from tailsitter.trim import AeroData, load_aero_data, R2D

# Skip if data files don't exist
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DATA_EXISTS = (DATA_DIR / "aero_cfg2_alpha.npy").exists()


@pytest.fixture
def aero():
    """Load real aerodynamic data."""
    if not DATA_EXISTS:
        pytest.skip("Processed data files not found")
    return load_aero_data("aero_cfg2", DATA_DIR)


@pytest.fixture
def sin4_coeffs(aero):
    """Build sin4 coefficients from real aero data."""
    return build_sin4_coeffs(aero)


@pytest.fixture
def mock_aero():
    """Create mock aerodynamic data."""
    alpha = np.linspace(-180, 180, 361)
    return AeroData(
        alpha=alpha,
        cL=np.sin(np.radians(alpha)) * 0.8,
        cD=np.abs(np.sin(np.radians(alpha))) * 0.1 + 0.02,
        cm=np.sin(np.radians(alpha) * 2) * 0.1,
        ele=np.array([-30, -20, -10, 0, 10, 20]),
        dcL=np.array([-0.2, -0.1, -0.02, 0.02, 0.1, 0.2]),
        dcD=np.array([-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]),
        dcm=np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]),
        throttle=np.linspace(0, 1, 11),
        thrust=np.linspace(0, 7, 11),
    )


# ---------------------------------------------------------------------------
# Sin4 model tests
# ---------------------------------------------------------------------------

class TestSin4Fit:
    """Test sinusoidal aero coefficient fitting."""

    def test_fit_sin4_shape(self):
        """fit_sin4 should return 12 coefficients."""
        x = np.linspace(-10, 20, 100)
        y = np.sin(x * np.pi / 180) * 0.5
        coeffs = fit_sin4(x, y)
        assert len(coeffs) == 12

    def test_fit_sin4_produces_finite_output(self):
        """fit_sin4 should produce finite coefficients that evaluate to finite values."""
        if not DATA_EXISTS:
            pytest.skip("Processed data files not found")
        aero = load_aero_data("aero_cfg2", DATA_DIR)
        x = aero.alpha
        y = aero.cL
        coeffs = fit_sin4(x, y)
        # Coefficients should be finite
        assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
        # Evaluate on a subset and check finiteness
        x_test = np.linspace(x.min(), x.max(), 50)
        y_fit = np.zeros_like(x_test)
        for i in range(4):
            a, b, c = coeffs[3 * i], coeffs[3 * i + 1], coeffs[3 * i + 2]
            y_fit += a * np.sin(b * x_test + c)
        assert np.all(np.isfinite(y_fit)), "Fit output should be finite"

    def test_build_sin4_coeffs(self, aero):
        """build_sin4_coeffs should produce valid coefficients."""
        coeffs = build_sin4_coeffs(aero)
        assert isinstance(coeffs, AeroSin4Coeffs)
        assert len(coeffs.cz) == 12
        assert len(coeffs.cL) == 12
        assert len(coeffs.cD) == 12
        assert len(coeffs.cm) == 12

    def test_load_sin4_coeffs(self):
        """load_sin4_coeffs should work with real data."""
        if not DATA_EXISTS:
            pytest.skip("Processed data files not found")
        coeffs = load_sin4_coeffs("aero_cfg2", DATA_DIR)
        assert isinstance(coeffs, AeroSin4Coeffs)
        assert len(coeffs.cz) == 12


class TestSin4Eval:
    """Test sin4 model evaluation with derivatives."""

    def test_eval_returns_tuple(self, sin4_coeffs):
        """aero_sin4_eval should return (fval, fval_dot, fval_ddot)."""
        result = aero_sin4_eval(0.1, sin4_coeffs.cz)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_eval_at_zero(self, sin4_coeffs):
        """aero_sin4_eval should work at alpha=0."""
        fval, fval_dot, fval_ddot = aero_sin4_eval(0.0, sin4_coeffs.cL)
        assert np.isfinite(fval)
        assert np.isfinite(fval_dot)
        assert np.isfinite(fval_ddot)

    def test_derivative_consistency(self, sin4_coeffs):
        """Numerical derivative should match analytical derivative."""
        alpha = 0.15  # rad
        h = 1e-5

        f_plus, _, _ = aero_sin4_eval(alpha + h, sin4_coeffs.cL)
        f_minus, _, _ = aero_sin4_eval(alpha - h, sin4_coeffs.cL)
        numerical_dot = (f_plus - f_minus) / (2 * h)

        _, analytical_dot, _ = aero_sin4_eval(alpha, sin4_coeffs.cL)

        assert abs(numerical_dot - analytical_dot) < 0.1, (
            f"Derivative mismatch: numerical={numerical_dot:.6f}, "
            f"analytical={analytical_dot:.6f}"
        )


# ---------------------------------------------------------------------------
# Differential flatness transfer tests
# ---------------------------------------------------------------------------

class TestDifferentialFlatnessTransfer:
    """Test the main DF transfer function."""

    def test_forward_flight(self, aero, sin4_coeffs):
        """DF transfer at forward flight conditions."""
        result = differential_flatness_transfer(
            V=15.0, Vdot=0.0, Vddot=0.0, Vdddot=0.0,
            gamma=np.radians(5.0), gammadot=0.0, gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero,
        )
        assert isinstance(result, DFTransferResult)
        assert np.isfinite(result.alpha)
        assert np.isfinite(result.theta)
        assert np.isfinite(result.q)
        assert np.isfinite(result.T)
        assert np.isfinite(result.elevator)
        assert abs(result.residual) < 0.01, f"Residual too large: {result.residual}"

    def test_hover_conditions(self, aero, sin4_coeffs):
        """DF transfer near hover."""
        result = differential_flatness_transfer(
            V=0.5, Vdot=0.0, Vddot=0.0, Vdddot=0.0,
            gamma=np.radians(85.0), gammadot=0.0, gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero,
        )
        assert np.isfinite(result.alpha)
        assert np.isfinite(result.T)
        assert result.T >= 0, "Thrust should be non-negative"

    def test_theta_equals_gamma_plus_alpha(self, aero, sin4_coeffs):
        """theta should equal gamma + alpha."""
        gamma = np.radians(30.0)
        result = differential_flatness_transfer(
            V=10.0, Vdot=0.0, Vddot=0.0, Vdddot=0.0,
            gamma=gamma, gammadot=0.0, gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero,
        )
        assert abs(result.theta - (gamma + result.alpha)) < 1e-6

    def test_with_acceleration(self, aero, sin4_coeffs):
        """DF transfer with nonzero Vdot."""
        result = differential_flatness_transfer(
            V=10.0, Vdot=1.0, Vddot=0.0, Vdddot=0.0,
            gamma=np.radians(20.0), gammadot=0.0, gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero,
        )
        assert np.isfinite(result.alpha)
        assert abs(result.residual) < 0.01

    def test_with_gamma_rate(self, aero, sin4_coeffs):
        """DF transfer with nonzero gammadot."""
        result = differential_flatness_transfer(
            V=12.0, Vdot=0.0, Vddot=0.0, Vdddot=0.0,
            gamma=np.radians(30.0), gammadot=np.radians(5.0),
            gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero,
        )
        assert np.isfinite(result.q)
        assert abs(result.q) > 0, "Pitch rate should be nonzero with gammadot"


# ---------------------------------------------------------------------------
# Pitch rate margin tests
# ---------------------------------------------------------------------------

class TestPitchRateMargin:
    """Test max feasible pitch rate calculation."""

    def test_returns_valid_margin(self, aero, sin4_coeffs):
        """Should return a valid PitchRateMargin."""
        margin = calculate_max_feasible_pitch_rate(
            V=10.0, gamma=np.radians(30.0),
            V_dot=0.0, gamma_dot=0.0,
            delta_e_flat=0.0, T_flat=30.0,
            alpha_flat=np.radians(5.0),
        )
        assert isinstance(margin, PitchRateMargin)
        assert margin.q_max >= 0, "q_max should be non-negative"
        assert margin.tau > 0, "tau should be positive"
        assert margin.eta >= 0, "eta should be non-negative"

    def test_custom_params(self):
        """Should accept custom PitchRateMarginParams."""
        params = PitchRateMarginParams(k=3.0, tau_min=0.1)
        margin = calculate_max_feasible_pitch_rate(
            V=10.0, gamma=np.radians(30.0),
            V_dot=0.0, gamma_dot=0.0,
            delta_e_flat=0.0, T_flat=30.0,
            alpha_flat=np.radians(5.0),
            params=params,
        )
        assert margin.tau >= 0.1

    def test_larger_margin_at_center(self):
        """Control margin should be larger when elevator is at center."""
        margin_center = calculate_max_feasible_pitch_rate(
            V=10.0, gamma=np.radians(30.0),
            V_dot=0.0, gamma_dot=0.0,
            delta_e_flat=5.0, T_flat=30.0,
            alpha_flat=np.radians(5.0),
            limit=(-20.0, 30.0),
        )
        margin_edge = calculate_max_feasible_pitch_rate(
            V=10.0, gamma=np.radians(30.0),
            V_dot=0.0, gamma_dot=0.0,
            delta_e_flat=25.0, T_flat=30.0,
            alpha_flat=np.radians(5.0),
            limit=(-20.0, 30.0),
        )
        assert margin_center.eta > margin_edge.eta
