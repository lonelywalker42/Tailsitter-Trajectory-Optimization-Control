"""Tests for trajectory optimization module."""

import numpy as np
import pytest
import casadi as cs

from tailsitter.trajectory_optimization import (
    TailsitterTrajectoryOptimizer,
    TrajectoryOptConfig,
    TrajectoryResult,
    _UAVParams,
)


@pytest.fixture
def config():
    """Default config with small node count for fast tests."""
    return TrajectoryOptConfig(num_nodes=50, ipopt_max_iter=2000)


@pytest.fixture
def optimizer(config):
    """Built optimizer with small problem."""
    opt = TailsitterTrajectoryOptimizer(config)
    opt.build()
    return opt


class TestTrajectoryOptConfig:
    def test_default_direction(self):
        cfg = TrajectoryOptConfig()
        assert cfg.direction == "forward2hover"

    def test_state_bounds_shape(self):
        cfg = TrajectoryOptConfig()
        assert cfg.x_min.shape == (11,)
        assert cfg.x_max.shape == (11,)

    def test_control_bounds_shape(self):
        cfg = TrajectoryOptConfig()
        assert cfg.u_min.shape == (3,)
        assert cfg.u_max.shape == (3,)


class TestInterpolants:
    def test_interpolants_built(self, config):
        opt = TailsitterTrajectoryOptimizer(config)
        assert opt._cL_interp is not None
        assert opt._thrust_interp is not None

    def test_cL_at_zero_alpha(self, config):
        """CL should be near -0.03 at alpha=0 (from cfg1 table)."""
        opt = TailsitterTrajectoryOptimizer(config)
        val = opt._cL_interp(0.0)
        assert abs(float(val) - (-0.0300427186306781)) < 1e-3

    def test_thrust_at_zero(self, config):
        """Throttle=0 should give zero thrust."""
        opt = TailsitterTrajectoryOptimizer(config)
        val = opt._thrust_interp(0.0)
        assert abs(float(val)) < 1e-6

    def test_thrust_at_one(self, config):
        """Throttle=1 should give ~7.053."""
        opt = TailsitterTrajectoryOptimizer(config)
        val = opt._thrust_interp(1.0)
        assert abs(float(val) - 7.053) < 0.1


class TestCollocationCoefficients:
    def test_coefficients_shape(self):
        C, D, B = TailsitterTrajectoryOptimizer._collocation_coefficients(1)
        assert C.shape == (2, 2)
        assert D.shape == (2,)
        assert B.shape == (2,)

    def test_D_sums_to_one(self):
        _, D, _ = TailsitterTrajectoryOptimizer._collocation_coefficients(1)
        assert abs(D.sum() - 1.0) < 1e-10

    def test_B_sums_to_one(self):
        _, _, B = TailsitterTrajectoryOptimizer._collocation_coefficients(1)
        assert abs(B.sum() - 1.0) < 1e-10

    def test_backward_euler_structure(self):
        """d=1 Radau should give backward Euler: C=[[-1,-1],[1,1]], D=[0,1]."""
        C, D, B = TailsitterTrajectoryOptimizer._collocation_coefficients(1)
        np.testing.assert_allclose(C, [[-1, -1], [1, 1]], atol=1e-10)
        np.testing.assert_allclose(D, [0, 1], atol=1e-10)
        np.testing.assert_allclose(B, [0.5, 0.5], atol=1e-10)


class TestDynamics:
    def test_ode_forward_flight(self, config):
        """Test dynamics at forward-flight trim condition."""
        opt = TailsitterTrajectoryOptimizer(config)

        # Forward flight: 11D state
        x = cs.DM([19.9264, 1.7147, 4.9184 * np.pi / 180, 0, 0,
                    0, 0, 0, 0.0572, 2.6443, 0])
        u_ctrl = cs.DM([0, 0, 0])

        xdot = opt._ode(x, u_ctrl)
        xdot_np = np.array(xdot).flatten()

        assert np.all(np.isfinite(xdot_np))
        # hdot should be near zero at trim
        assert abs(xdot_np[4]) < 5.0

    def test_ode_hover(self, config):
        """Test dynamics at hover trim condition."""
        opt = TailsitterTrajectoryOptimizer(config)

        x = cs.DM([0.01, 0, np.pi / 2, 0, 0,
                    0, 0, 0, 0.6339, 0, 0])
        u_ctrl = cs.DM([0, 0, 0])

        xdot = opt._ode(x, u_ctrl)
        xdot_np = np.array(xdot).flatten()

        assert np.all(np.isfinite(xdot_np))

    def test_ode_shape(self, config):
        """Output should be 11D."""
        opt = TailsitterTrajectoryOptimizer(config)
        x = cs.DM([10, 1, 0.1, 0, 5, 0, 0, 0, 0.5, 0, 0])
        u_ctrl = cs.DM([0.01, 0, 0])

        xdot = opt._ode(x, u_ctrl)
        assert xdot.shape == (11, 1)


class TestOptimizerBuild:
    def test_build_succeeds(self, config):
        opt = TailsitterTrajectoryOptimizer(config)
        opt.build()
        assert opt._solver is not None

    def test_solve_produces_result(self, optimizer):
        """Problem should solve and produce a result."""
        result = optimizer.solve()
        N = optimizer.cfg.num_nodes
        assert isinstance(result, TrajectoryResult)
        assert result.time.shape == (N + 1,)
        assert result.state.shape == (N + 1, 11)
        assert result.control.shape == (N, 3)
        assert np.isfinite(result.cost)
        assert result.tf > 0

    def test_boundary_conditions_satisfied(self, optimizer):
        """Initial and final states should match boundary conditions."""
        result = optimizer.solve()
        cfg = optimizer.cfg

        x0_actual = result.state[0, :]
        xf_actual = result.state[-1, :]
        x0_expected = cfg.x0_forward
        xf_expected = cfg.xf_hover

        np.testing.assert_allclose(x0_actual, x0_expected, atol=1e-3)
        np.testing.assert_allclose(xf_actual, xf_expected, atol=1e-3)


class TestTrajectoryResult:
    def test_dataclass_fields(self):
        result = TrajectoryResult(
            time=np.array([0, 1]),
            state=np.zeros((2, 11)),
            control=np.zeros((1, 3)),
            cost=10.0,
            solve_time=1.0,
            success=True,
            tf=1.0,
        )
        assert result.success
        assert result.tf == 1.0
