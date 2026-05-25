"""Tests for reward module."""

import numpy as np
import pytest

from tailsitter.config import EnvConfig
from tailsitter.reward import RewardCalculator


@pytest.fixture
def reward_calc():
    config = EnvConfig()
    return RewardCalculator(config)


def test_shaping_func_peak():
    assert RewardCalculator.shaping_func(0.0) == pytest.approx(1.0)


def test_shaping_func_decreases():
    assert RewardCalculator.shaping_func(0.0) > RewardCalculator.shaping_func(1.0)
    assert RewardCalculator.shaping_func(1.0) > RewardCalculator.shaping_func(10.0)


def test_shaping_func_positive():
    for x in [-10, -1, 0, 1, 10]:
        assert RewardCalculator.shaping_func(x) > 0


def test_reward_at_target(reward_calc):
    """At target state, reward should be high (r_V, r_theta, r_alpha ≈ 1)."""
    state = np.array([0.0, 0.0, 20.0, 0.086, 0.086, 0.0, 10.0])  # at target
    target = np.array([20.0, 0.086, 0.086])
    reward, _ = reward_calc.compute(state, target, t=0.0, success_integral=0.0)
    # r_V ≈ 1, r_theta ≈ 1, r_alpha ≈ 1, r_dh ≈ 0, time_coef ≈ -0.05
    # r_success starts at 0
    assert reward > 2.5  # ~3 - 0.05


def test_reward_increases_with_success_integral(reward_calc):
    state = np.array([0.0, 0.0, 20.0, 0.086, 0.086, 0.0, 10.0])
    target = np.array([20.0, 0.086, 0.086])
    r1, si1 = reward_calc.compute(state, target, t=0.0, success_integral=0.0)
    r2, si2 = reward_calc.compute(state, target, t=0.0, success_integral=0.5)
    assert r2 > r1  # higher success_integral -> higher reward


def test_success_integral_accumulates(reward_calc):
    """Success integral should accumulate when near target."""
    state = np.array([0.0, 0.0, 20.0, 0.086, 0.086, 0.0, 10.0])
    target = np.array([20.0, 0.086, 0.086])
    _, si = reward_calc.compute(state, target, t=0.0, success_integral=0.0)
    assert si > 0.0


def test_success_integral_clamps(reward_calc):
    state = np.array([0.0, 0.0, 20.0, 0.086, 0.086, 0.0, 10.0])
    target = np.array([20.0, 0.086, 0.086])
    _, si = reward_calc.compute(state, target, t=0.0, success_integral=0.99)
    assert si <= 1.0


def test_time_penalty_grows(reward_calc):
    state = np.array([0.0, 0.0, 20.0, 0.086, 0.086, 0.0, 10.0])
    target = np.array([20.0, 0.086, 0.086])
    r_early, _ = reward_calc.compute(state, target, t=1.0, success_integral=0.0)
    r_late, _ = reward_calc.compute(state, target, t=10.0, success_integral=0.0)
    # Time penalty grows: -0.05*exp(0.1*t)
    assert r_late < r_early
