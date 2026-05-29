"""Tests for DF-based trim corridor sweep."""

import numpy as np
import pytest
from pathlib import Path

from tailsitter.df_trim import (
    DFTrimPoint,
    DFTrimMesh,
    df_trim_sweep,
    save_df_trim_mesh,
    load_df_trim_mesh,
)

# Skip if data files don't exist
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DATA_EXISTS = (DATA_DIR / "aero_cfg2_alpha.npy").exists()


@pytest.fixture
def small_mesh():
    """Run a small DF trim sweep for testing."""
    if not DATA_EXISTS:
        pytest.skip("Processed data files not found")
    return df_trim_sweep(
        V_range=(5.0, 15.0),
        gamma_range=(10.0, 60.0),
        dV=2.0,
        dgamma=10.0,
        config_name="aero_cfg2",
        data_dir=DATA_DIR,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Data container tests
# ---------------------------------------------------------------------------

class TestDFTrimPoint:
    """Test DFTrimPoint dataclass."""

    def test_default_values(self):
        """Default values should be zero/False."""
        pt = DFTrimPoint()
        assert pt.alpha == 0.0
        assert pt.T == 0.0
        assert pt.trim_flag is False
        assert pt.residual == 0.0


class TestDFTrimMesh:
    """Test DFTrimMesh dataclass."""

    def test_default_arrays(self):
        """Default arrays should be empty."""
        mesh = DFTrimMesh()
        assert len(mesh.trim_V) == 0
        assert len(mesh.V_grid) == 0


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------

class TestDFTriMTrimSweep:
    """Test df_trim_sweep function."""

    def test_grid_shapes(self, small_mesh):
        """Grids should have correct shapes."""
        n_gamma = len(small_mesh.trim_gamma)
        n_V = len(small_mesh.trim_V)
        assert small_mesh.V_grid.shape == (n_gamma, n_V)
        assert small_mesh.gamma_grid.shape == (n_gamma, n_V)
        assert small_mesh.flag_grid.shape == (n_gamma, n_V)
        assert small_mesh.alpha_grid.shape == (n_gamma, n_V)
        assert small_mesh.theta_grid.shape == (n_gamma, n_V)
        assert small_mesh.q_grid.shape == (n_gamma, n_V)
        assert small_mesh.T_grid.shape == (n_gamma, n_V)
        assert small_mesh.ele_grid.shape == (n_gamma, n_V)
        assert small_mesh.qmax_grid.shape == (n_gamma, n_V)

    def test_feasibility_flags(self, small_mesh):
        """Feasibility flags should be 0 or 1."""
        unique_vals = np.unique(small_mesh.flag_grid)
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_has_feasible_points(self, small_mesh):
        """Should have at least some feasible points."""
        n_feasible = int(np.sum(small_mesh.flag_grid))
        assert n_feasible > 0, "Should have at least some feasible points"

    def test_qmax_nonnegative_at_feasible(self, small_mesh):
        """qmax should be non-negative at feasible points."""
        feasible = small_mesh.flag_grid > 0
        qmax_feasible = small_mesh.qmax_grid[feasible]
        assert np.all(qmax_feasible >= 0), "qmax should be non-negative at feasible points"

    def test_thrust_nonnegative_at_feasible(self, small_mesh):
        """Thrust should be non-negative at feasible points."""
        feasible = small_mesh.flag_grid > 0
        T_feasible = small_mesh.T_grid[feasible]
        assert np.all(T_feasible >= 0), "Thrust should be non-negative at feasible points"


# ---------------------------------------------------------------------------
# Save/load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Test save/load round-trip."""

    def test_round_trip(self, small_mesh, tmp_path):
        """Save then load should produce identical data."""
        path = tmp_path / "test_mesh.npz"
        save_df_trim_mesh(small_mesh, path)
        assert path.exists()

        loaded = load_df_trim_mesh(path)
        np.testing.assert_array_equal(loaded.trim_V, small_mesh.trim_V)
        np.testing.assert_array_equal(loaded.trim_gamma, small_mesh.trim_gamma)
        np.testing.assert_array_equal(loaded.V_grid, small_mesh.V_grid)
        np.testing.assert_array_equal(loaded.flag_grid, small_mesh.flag_grid)
        np.testing.assert_array_equal(loaded.alpha_grid, small_mesh.alpha_grid)
        np.testing.assert_array_equal(loaded.qmax_grid, small_mesh.qmax_grid)
        assert loaded.Vdot == small_mesh.Vdot
        assert loaded.gammadot == small_mesh.gammadot
