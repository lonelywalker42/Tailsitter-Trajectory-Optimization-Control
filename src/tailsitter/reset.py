"""Episode reset logic for different flight modes.

Replaces the MATLAB localResetFcn in train_script.m and trian_PPO_script.m.
"""

import numpy as np

from .config import ResetConfig


def _uw_to_Valpha(u: float, w: float) -> tuple[float, float]:
    """Compute V and alpha from body-axis velocities u, w."""
    V = np.sqrt(u**2 + w**2)
    if V > 1e-10:
        alpha = np.arctan2(w, u)
    else:
        alpha = np.pi / 2
    return V, alpha


def _expand_state(initial_5d: np.ndarray) -> np.ndarray:
    """Convert [u, w, theta, q, h] to [u, w, V, alpha, theta, q, h]."""
    u, w, theta, q, h = initial_5d
    V, alpha = _uw_to_Valpha(u, w)
    return np.array([u, w, V, alpha, theta, q, h])


class EpisodeResetter:
    """Manages episode initialization for different flight modes."""

    def __init__(self, config: ResetConfig):
        self.cfg = config

    def reset_sac(self) -> tuple[np.ndarray, np.ndarray, int]:
        """SAC reset: 99% back-transition, 1% forward with randomized state."""
        if np.random.random() < self.cfg.sac_forward_prob:
            cmd = -1
            target = np.array(self.cfg.forward_state)
            base_init = np.array(self.cfg.hover_initial)
            forward_init = np.array(self.cfg.forward_initial)
            frac = np.random.random() / 2.0
            initial_state = _expand_state(base_init + frac * (forward_init - base_init))
        else:
            cmd = 1
            target = np.array(self.cfg.hover_state)
            initial_state = _expand_state(np.array(self.cfg.forward_initial))
        return initial_state, target, cmd

    def reset_ppo(self) -> tuple[np.ndarray, np.ndarray, int]:
        """PPO reset: 50/50, deterministic initial states."""
        if np.random.random() < self.cfg.ppo_forward_prob:
            cmd = -1
            target = np.array(self.cfg.forward_state)
            initial_state = _expand_state(np.array(self.cfg.hover_initial))
        else:
            cmd = 1
            target = np.array(self.cfg.hover_state)
            initial_state = _expand_state(np.array(self.cfg.forward_initial))
        return initial_state, target, cmd
