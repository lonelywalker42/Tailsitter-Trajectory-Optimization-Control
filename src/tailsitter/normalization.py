"""State and action normalization utilities.

Matches the Simulink Gain blocks in LonEnv.slx:
- Observation: obs = NormMatrixState @ raw_obs
- Action denorm: physical = inv(NormMatrixAct) @ action + BiasAct
"""

import numpy as np

from .config import NormalizationConfig


class Normalizer:
    """Handles state normalization and action denormalization."""

    def __init__(self, config: NormalizationConfig):
        self.norm_state = config.state_norm_matrix       # 11x11 diagonal
        self.inv_norm_act = config.action_inv_norm_matrix # 2x2 diagonal
        self.bias_act = config.action_bias                # 2x1

    def normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize observation: obs = NormMatrixState @ raw_obs."""
        return self.norm_state @ raw_obs

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert [-1,1] agent action to physical [thrust, elevator]."""
        return self.inv_norm_act @ action + self.bias_act
