"""Tests for normalization module."""

import numpy as np
import pytest

from tailsitter.config import NormalizationConfig
from tailsitter.normalization import Normalizer


@pytest.fixture
def normalizer():
    config = NormalizationConfig()
    return Normalizer(config)


def test_normalize_obs_shape(normalizer):
    raw = np.zeros(11)
    result = normalizer.normalize_obs(raw)
    assert result.shape == (11,)


def test_normalize_obs_identity_at_zero(normalizer):
    raw = np.zeros(11)
    result = normalizer.normalize_obs(raw)
    np.testing.assert_array_almost_equal(result, np.zeros(11))


def test_normalize_obs_scaling(normalizer):
    raw = np.ones(11)
    result = normalizer.normalize_obs(raw)
    expected = np.array([0.1, 0.05, 0.05, 0.1, 0.5, 1.0, 0.05, 0.2, 0.1, 0.1, 1.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_denormalize_action_shape(normalizer):
    action = np.array([0.0, 0.0])
    result = normalizer.denormalize_action(action)
    assert result.shape == (2,)


def test_denormalize_action_midpoint(normalizer):
    action = np.array([0.0, 0.0])
    result = normalizer.denormalize_action(action)
    # physical = inv(diag([2, 0.04])) @ [0, 0] + [-0.5, 5]
    expected = np.array([-0.5, 5.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_denormalize_action_range(normalizer):
    action_min = np.array([-1.0, -1.0])
    action_max = np.array([1.0, 1.0])
    phys_min = normalizer.denormalize_action(action_min)
    phys_max = normalizer.denormalize_action(action_max)
    # physical = inv(diag([2, 0.04])) @ action + [-0.5, 5]
    # dt: [-1,1] -> [0.5*(-1)-0.5, 0.5*(1)-0.5] = [-1.0, 0.0]
    # de: [-1,1] -> [25*(-1)+5, 25*(1)+5] = [-20.0, 30.0]
    np.testing.assert_almost_equal(phys_min[0], -1.0)
    np.testing.assert_almost_equal(phys_max[0], 0.0)
    np.testing.assert_almost_equal(phys_min[1], -20.0)
    np.testing.assert_almost_equal(phys_max[1], 30.0)
