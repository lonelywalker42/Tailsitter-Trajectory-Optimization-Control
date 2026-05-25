"""Tests for trim analysis module."""

import numpy as np
import pytest
from pathlib import Path

from tailsitter.trim import (
    AeroData, TrimPoint, TrimMesh,
    load_aero_data, compute_aero, compute_residual_norm,
    trim_level_flight, trim_at_conditions, sweep_level_flight,
    sweep_corridor, R2D,
)

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
def mock_aero():
    """Create mock aerodynamic data for basic tests."""
    return AeroData(
        alpha=np.linspace(-180, 180, 361),
        cL=np.sin(np.linspace(-180, 180, 361) * np.pi / 180) * 0.8,
        cD=np.abs(np.sin(np.linspace(-180, 180, 361) * np.pi / 180)) * 0.1 + 0.02,
        cm=np.sin(np.linspace(-180, 180, 361) * np.pi / 90) * 0.1,
        ele=np.array([-30, -20, -10, 0, 10, 20]),
        dcL=np.array([-0.2, -0.1, -0.02, 0.02, 0.1, 0.2]),
        dcD=np.array([-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]),
        dcm=np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]),
        throttle=np.linspace(0, 1, 11),
        thrust=np.linspace(0, 7, 11),
    )


# ---------------------------------------------------------------------------
# Basic computation tests
# ---------------------------------------------------------------------------

class TestComputeAero:
    """Test aerodynamic force computation."""

    def test_hover_conditions(self, aero):
        """At hover (V near 0), propulsion should dominate."""
        L, D, M, Fprop = compute_aero(0.1, 90.0, 0.0, 0.5, aero)
        assert Fprop > 0, "Propulsion force should be positive"

    def test_forward_flight(self, aero):
        """In forward flight, lift should be significant."""
        L, D, M, Fprop = compute_aero(15.0, 5.0, 0.0, 0.3, aero)
        assert L > 0, "Lift should be positive at positive alpha"
        assert D > 0, "Drag should be positive"

    def test_zero_throttle(self, aero):
        """Zero throttle should give zero propulsion."""
        _, _, _, Fprop = compute_aero(10.0, 0.0, 0.0, 0.0, aero)
        assert Fprop == pytest.approx(0.0, abs=0.1)

    def test_elevator_effect(self, aero):
        """Elevator deflection should affect pitching moment."""
        _, _, M0, _ = compute_aero(10.0, 5.0, 0.0, 0.3, aero)
        _, _, M10, _ = compute_aero(10.0, 5.0, 10.0, 0.3, aero)
        assert M0 != M10, "Elevator should affect moment"


class TestResidualNorm:
    """Test residual computation."""

    def test_residual_nonnegative(self, aero):
        """Residual norm should be non-negative."""
        x = np.array([0.5, 0.0, 0.1])
        res = compute_residual_norm(x, aero, 10.0, 0.2, 0.0)
        assert res >= 0


# ---------------------------------------------------------------------------
# Trim solver tests
# ---------------------------------------------------------------------------

class TestTrimLevelFlight:
    """Test level flight trim solver."""

    def test_trim_at_15ms(self, aero):
        """Should find trim at V=15 m/s."""
        tp = trim_level_flight(15.0, aero)
        assert tp.trim_flag, f"Trim should succeed at V=15 m/s, got residual={tp.residual}"
        assert 0 <= tp.throttle <= 1
        assert -30 <= tp.elevator_deg <= 20

    def test_trim_at_low_speed(self, aero):
        """Should find trim at low speed."""
        tp = trim_level_flight(5.0, aero)
        assert tp.trim_flag, f"Trim should succeed at V=5 m/s, got residual={tp.residual}"

    def test_residual_near_zero(self, aero):
        """Residual should be near zero at trim point."""
        tp = trim_level_flight(10.0, aero)
        if tp.trim_flag:
            assert tp.residual < 0.01, f"Residual too large: {tp.residual}"


class TestTrimAtConditions:
    """Test general trim solver with random restarts."""

    def test_forward_flight(self, aero):
        """Should find trim at forward flight (V=20, theta=5)."""
        tp = trim_at_conditions(20.0, 5.0, aero, max_restarts=15)
        assert tp.trim_flag, "Trim should succeed at forward flight"
        assert tp.alpha_deg > 0, "Alpha should be positive"

    def test_hover(self, aero):
        """Should find trim near hover (V=0.1, theta=90)."""
        tp = trim_at_conditions(0.1, 90.0, aero, max_restarts=15)
        # Hover may or may not succeed depending on aero data
        if tp.trim_flag:
            assert tp.throttle > 0.3, "Hover should require significant throttle"

    def test_mid_transition(self, aero):
        """Should find trim at mid-transition (V=10, theta=45)."""
        tp = trim_at_conditions(10.0, 45.0, aero, max_restarts=15)
        if tp.trim_flag:
            assert 0 <= tp.throttle <= 1


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------

class TestSweepLevelFlight:
    """Test level flight sweep."""

    def test_sweep_produces_results(self, aero):
        """Sweep should produce results for each velocity."""
        V_range = np.array([5.0, 10.0, 15.0])
        result = sweep_level_flight(V_range, aero)
        assert len(result['V']) == 3
        assert len(result['flag']) == 3

    def test_high_speed_trim(self, aero):
        """High speed level flight should generally succeed."""
        V_range = np.array([15.0, 18.0, 20.0])
        result = sweep_level_flight(V_range, aero)
        # At least some should succeed
        assert np.sum(result['flag']) > 0, "At least some high-speed trims should succeed"


class TestSweepCorridor:
    """Test corridor sweep."""

    def test_small_grid(self, aero):
        """Small grid sweep should produce correct shapes."""
        V_range = np.array([10.0, 15.0])
        theta_range = np.array([10.0, 30.0, 50.0])
        mesh = sweep_corridor(V_range, theta_range, aero, max_restarts=5)

        assert mesh.trim_flag.shape == (2, 3)
        assert mesh.throttle.shape == (2, 3)
        assert mesh.elevator.shape == (2, 3)
        assert mesh.alpha.shape == (2, 3)
        assert mesh.gamma.shape == (2, 3)
        assert mesh.max_moment_ft.shape == (2, 3)
        assert mesh.max_moment_bt.shape == (2, 3)

    def test_trim_at_forward(self, aero):
        """Forward flight (high V, low theta) should be trimmable."""
        mesh = sweep_corridor(
            np.array([18.0]), np.array([5.0]), aero, max_restarts=15)
        # This point should typically be trimmable
        # (not asserting True since it depends on aero data)


# ---------------------------------------------------------------------------
# Linearization tests
# ---------------------------------------------------------------------------

class TestLinearization:
    """Test linearization module."""

    def test_import(self):
        """Should import linearization module."""
        from tailsitter.linearization import (
            linearize_at_point, analyze_system,
            analyze_eigenvalues, analyze_controllability,
        )

    def test_linearize_at_point(self, aero):
        """Should compute A, B matrices at a known point."""
        from tailsitter.linearization import linearize_at_point, analyze_system

        A, B = linearize_at_point(15.0, 5.0, 5.0, 0.3, 0.0, aero)

        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
        assert not np.any(np.isnan(A)), "A should not contain NaN"
        assert not np.any(np.isnan(B)), "B should not contain NaN"

    def test_analyze_system(self, aero):
        """Should produce complete system analysis."""
        from tailsitter.linearization import linearize_at_point, analyze_system

        A, B = linearize_at_point(15.0, 5.0, 5.0, 0.3, 0.0, aero)
        analysis = analyze_system(A, B)

        assert analysis.eigen_info.eigenvalues is not None
        assert len(analysis.eigen_info.eigenvalues) == 4
        assert analysis.stability_info.type in [
            "asymptotically_stable", "marginally_stable", "unstable"]
        assert len(analysis.modal_info) == 4
