"""Tests for Gymnasium environment."""

import numpy as np
import pytest

from tailsitter.config import (
    EnvConfig,
    NormalizationConfig,
    PhysicalConfig,
    ResetConfig,
)
from tailsitter.dynamics import LongitudinalDynamics
from tailsitter.env import TailsitterLongitudinalEnv
from tailsitter.normalization import Normalizer
from tailsitter.reset import EpisodeResetter
from tailsitter.reward import RewardCalculator


@pytest.fixture
def mock_aero_data():
    """Create mock aerodynamic data for testing."""
    alpha = np.linspace(-180, 180, 41).reshape(-1, 1).astype(np.float64)
    return {
        "alpha": alpha,
        "cL": np.sin(np.radians(alpha)) * 0.8,
        "cD": np.abs(np.sin(np.radians(alpha))) * 0.1 + 0.02,
        "cm": np.sin(np.radians(alpha * 2)) * 0.1,
        "ele": np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]).reshape(-1, 1).astype(np.float64),
        "dcL": np.linspace(-0.3, 0.15, 11).reshape(-1, 1),
        "dcD": np.linspace(-0.005, 0.05, 11).reshape(-1, 1),
        "dcm": np.linspace(-0.04, 0.06, 11).reshape(-1, 1),
        "throttle": np.linspace(0, 1, 12).reshape(1, -1),
        "thrust": np.linspace(0, 7, 12).reshape(1, -1),
    }


@pytest.fixture
def env(mock_aero_data):
    phys = PhysicalConfig()
    env_cfg = EnvConfig()
    norm_cfg = NormalizationConfig()
    reset_cfg = ResetConfig()

    dynamics = LongitudinalDynamics(phys, mock_aero_data)
    reward_calc = RewardCalculator(env_cfg)
    normalizer = Normalizer(norm_cfg)
    resetter = EpisodeResetter(reset_cfg)

    return TailsitterLongitudinalEnv(
        dynamics=dynamics,
        reward_calc=reward_calc,
        normalizer=normalizer,
        resetter=resetter,
        env_config=env_cfg,
        phys_config=phys,
        reset_mode="sac",
    )


def test_observation_space_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (11,)


def test_action_space_shape(env):
    assert env.action_space.shape == (2,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_reset_returns_observation(env):
    obs, info = env.reset()
    assert obs.shape == (11,)
    assert isinstance(info, dict)


def test_step_returns_correct_types(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_observation_is_normalized(env):
    """Observation should be normalized (scaled by NormMatrixState)."""
    obs, _ = env.reset()
    # Values should be scaled (not raw physics values)
    assert np.all(np.abs(obs) < 100)  # reasonable range


def test_termination_on_ground_crash(env):
    """Should terminate when h <= -20."""
    env.reset()
    env._state[6] = -25.0  # set h below -20
    terminated = env._check_terminated()
    assert terminated


def test_termination_on_high_altitude(env):
    """Should terminate when h >= 50."""
    env.reset()
    env._state[6] = 55.0
    terminated = env._check_terminated()
    assert terminated


def test_termination_on_extreme_alpha(env):
    """Should terminate when alpha_deg >= 120."""
    env.reset()
    env._state[3] = np.radians(125.0)
    terminated = env._check_terminated()
    assert terminated


def test_termination_on_theta_25(env):
    """Should terminate when theta_deg >= 25."""
    env.reset()
    env._state[4] = np.radians(30.0)
    terminated = env._check_terminated()
    assert terminated


def test_no_termination_at_forward_flight(env):
    """Forward flight state should not trigger termination."""
    env.reset()
    env._state = np.array([19.926, 1.715, 20.0, 4.9184 * np.pi / 180,
                           4.9184 * np.pi / 180, 0.0, 10.0])
    env._success_integral = 0.0
    terminated = env._check_terminated()
    assert not terminated
